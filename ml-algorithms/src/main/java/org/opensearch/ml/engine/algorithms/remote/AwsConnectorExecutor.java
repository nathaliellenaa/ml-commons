/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.remote;

import static org.opensearch.ml.common.connector.ConnectorProtocols.AWS_SIGV4;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.LLM_INTERFACE_BEDROCK_CONVERSE_CLAUDE;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.LLM_INTERFACE;
import static software.amazon.awssdk.http.SdkHttpMethod.GET;
import static software.amazon.awssdk.http.SdkHttpMethod.POST;

import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.commons.text.StringEscapeUtils;
import org.apache.logging.log4j.Logger;
import org.opensearch.arrow.spi.StreamManager;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.util.TokenBucket;
import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.connector.AwsConnector;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.model.MLGuard;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.ml.engine.annotation.ConnectorExecutor;
import org.opensearch.ml.engine.httpclient.MLHttpClientFactory;
import org.opensearch.script.ScriptService;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.StreamTransportService;
import org.opensearch.transport.client.Client;

import com.jayway.jsonpath.JsonPath;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import okhttp3.OkHttpClient;
import okhttp3.Response;
import okhttp3.internal.http2.StreamResetException;
import okhttp3.sse.EventSource;
import okhttp3.sse.EventSourceListener;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.AwsSessionCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.internal.http.async.SimpleHttpContentPublisher;
import software.amazon.awssdk.http.SdkHttpFullRequest;
import software.amazon.awssdk.http.async.AsyncExecuteRequest;
import software.amazon.awssdk.http.async.SdkAsyncHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeAsyncClient;
import software.amazon.awssdk.services.bedrockruntime.model.ContentBlock;
import software.amazon.awssdk.services.bedrockruntime.model.ContentBlockDeltaEvent;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseStreamRequest;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseStreamResponseHandler;
import software.amazon.awssdk.services.bedrockruntime.model.Message;

@Log4j2
@ConnectorExecutor(AWS_SIGV4)
public class AwsConnectorExecutor extends AbstractConnectorExecutor {

    @Getter
    private AwsConnector connector;
    @Setter
    @Getter
    private ScriptService scriptService;
    @Setter
    @Getter
    private TokenBucket rateLimiter;
    @Setter
    @Getter
    private Map<String, TokenBucket> userRateLimiterMap;
    @Setter
    @Getter
    private Client client;
    @Setter
    @Getter
    private MLGuard mlGuard;

    private SdkAsyncHttpClient httpClient;

    @Setter
    @Getter
    private StreamManager streamManager;
    // Keep standard SSE client with MIME type text/event-stream for future references like SageMaker hosting models etc.
    private OkHttpClient okHttpClient;
    // Introduce bedrock client, as bedrock streaming services only support the specific MIME type application/vnd.amazon.eventstream.
    private BedrockRuntimeAsyncClient bedrockRuntimeAsyncClient;
    @Setter
    @Getter
    private ThreadPool threadPool;

    @Setter
    @Getter
    private StreamTransportService streamTransportService;

    private AtomicBoolean isStreamClosed = new AtomicBoolean(false);

    public AwsConnectorExecutor(Connector connector) {
        super.initialize(connector);
        this.connector = (AwsConnector) connector;
        Duration connectionTimeout = Duration.ofSeconds(super.getConnectorClientConfig().getConnectionTimeout());
        Duration readTimeout = Duration.ofSeconds(super.getConnectorClientConfig().getReadTimeout());
        Integer maxConnection = super.getConnectorClientConfig().getMaxConnections();
        this.httpClient = MLHttpClientFactory.getAsyncHttpClient(connectionTimeout, readTimeout, maxConnection);
        this.bedrockRuntimeAsyncClient = buildBedrockRuntimeAsyncClient(httpClient);
        try {
            AccessController.doPrivileged((PrivilegedExceptionAction<Void>) () -> {
                this.okHttpClient = new OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .readTimeout(1, TimeUnit.MINUTES)
                    .retryOnConnectionFailure(true)
                    .build();
                return null;
            });
        } catch (PrivilegedActionException e) {
            throw new RuntimeException("Failed to build OkHttpClient.", e);
        }
    }

    @Override
    public Logger getLogger() {
        return log;
    }

    @SuppressWarnings("removal")
    @Override
    public void invokeRemoteService(
        String action,
        MLInput mlInput,
        Map<String, String> parameters,
        String payload,
        ExecutionContext executionContext,
        ActionListener<Tuple<Integer, ModelTensors>> actionListener
    ) {
        try {
            SdkHttpFullRequest request;
            switch (connector.getActionHttpMethod(action).toUpperCase(Locale.ROOT)) {
                case "POST":
                    log.debug("original payload to remote model: " + payload);
                    request = ConnectorUtils.buildSdkRequest(action, connector, parameters, payload, POST);
                    break;
                case "GET":
                    request = ConnectorUtils.buildSdkRequest(action, connector, parameters, null, GET);
                    break;
                default:
                    throw new IllegalArgumentException("unsupported http method");
            }
            AsyncExecuteRequest executeRequest = AsyncExecuteRequest
                .builder()
                .request(signRequest(request))
                .requestContentPublisher(new SimpleHttpContentPublisher(request))
                .responseHandler(
                    new MLSdkAsyncHttpResponseHandler(
                        executionContext,
                        actionListener,
                        parameters,
                        connector,
                        scriptService,
                        mlGuard,
                        action
                    )
                )
                .build();
            AccessController.doPrivileged((PrivilegedExceptionAction<CompletableFuture<Void>>) () -> httpClient.execute(executeRequest));
        } catch (RuntimeException exception) {
            log.error("Failed to execute {} in aws connector: {}", action, exception.getMessage(), exception);
            actionListener.onFailure(exception);
        } catch (Throwable e) {
            log.error("Failed to execute {} in aws connector", action, e);
            actionListener.onFailure(new MLException("Fail to execute " + action + " in aws connector", e));
        }
    }

    @Override
    public void invokeRemoteServiceStream(
        String action,
        MLInput mlInput,
        Map<String, String> parameters,
        String payload,
        ExecutionContext executionContext,
        StreamPredictActionListener<MLTaskResponse, ?> actionListener
    ) {
        try {
            String llmInterface = parameters.get(LLM_INTERFACE);
            llmInterface = llmInterface.trim().toLowerCase(Locale.ROOT);
            llmInterface = StringEscapeUtils.unescapeJava(llmInterface);
            validateLLMInterface(llmInterface);

            ConverseStreamRequest request = ConverseStreamRequest
                .builder()
                .modelId(parameters.get("model"))
                .messages(Message.builder().role("user").content(ContentBlock.builder().text(parameters.get("inputs")).build()).build())
                .build();

            ConverseStreamResponseHandler handler = ConverseStreamResponseHandler.builder().onResponse(response -> {
                // Handle initial response
                getLogger().debug("Initial converse stream response: {}", response);
            }).onError(error -> {
                // Handle errors
                getLogger().error("Converse stream error: {}", error.getMessage());
            }).onComplete(() -> {
                // Handle completion
                getLogger().debug("Converse stream complete");
                sendCompletionResponse(actionListener);
            }).subscriber(event -> {
                getLogger().debug("Converse stream event: {}", event);
                switch (event.sdkEventType()) {
                    case CONTENT_BLOCK_DELTA:
                        ContentBlockDeltaEvent contentEvent = (ContentBlockDeltaEvent) event;
                        String chunk = contentEvent.delta().text();
                        sendContentResponse(chunk, false, actionListener);
                        break;
                    default:
                        // Ignore the other event types for now.
                        break;
                }
            }).build();
            bedrockRuntimeAsyncClient.converseStream(request, handler);
        } catch (Exception e) {
            log.error("Failed to execute streaming", e);
            actionListener.onFailure(new MLException("Fail to execute streaming", e));
        }
    }

    private void sendContentResponse(String content, boolean isLast, StreamPredictActionListener<MLTaskResponse, ?> streamActionListener) {
        log.info("sendContentResponse called with content: '{}', isLast: {}", content, isLast);
        List<ModelTensor> modelTensors = new ArrayList<>();
        Map<String, Object> dataMap = Map.of("content", content, "is_last", isLast);

        modelTensors.add(ModelTensor.builder().name("response").dataAsMap(dataMap).build());
        ModelTensorOutput output = ModelTensorOutput
            .builder()
            .mlModelOutputs(List.of(ModelTensors.builder().mlModelTensors(modelTensors).build()))
            .build();
        MLTaskResponse response = MLTaskResponse.builder().output(output).build();
        log.info("Calling streamActionListener.onStreamResponse with isLast: {}", isLast);
        streamActionListener.onStreamResponse(response, isLast);
    }

    private void sendCompletionResponse(StreamPredictActionListener<MLTaskResponse, ?> streamActionListener) {
        log.info("Sending completion response");
        if (isStreamClosed.compareAndSet(false, true)) {
            sendContentResponse("", true, streamActionListener);
        }
    }

    private BedrockRuntimeAsyncClient buildBedrockRuntimeAsyncClient(SdkAsyncHttpClient sdkAsyncHttpClient) {
        AwsSessionCredentials credentials = AwsSessionCredentials
            .create(connector.getAccessKey(), connector.getSecretKey(), connector.getSessionToken());
        AwsCredentialsProvider awsCredentialsProvider = StaticCredentialsProvider.create(credentials);

        return BedrockRuntimeAsyncClient
            .builder()
            .region(Region.of(connector.getRegion()))
            .credentialsProvider(awsCredentialsProvider)
            .httpClient(sdkAsyncHttpClient)
            .build();
    }

    private SdkHttpFullRequest signRequest(SdkHttpFullRequest request) {
        String accessKey = connector.getAccessKey();
        String secretKey = connector.getSecretKey();
        String sessionToken = connector.getSessionToken();
        String signingName = connector.getServiceName();
        String region = connector.getRegion();

        return ConnectorUtils.signRequest(request, accessKey, secretKey, sessionToken, signingName, region);
    }

    private void validateLLMInterface(String llmInterface) {
        switch (llmInterface) {
            case LLM_INTERFACE_BEDROCK_CONVERSE_CLAUDE:
                break;
            default:
                throw new MLException(String.format("Unsupported llm interface: %s", llmInterface));
        }
    }

    public final class AwsEventSourceListener extends EventSourceListener {
        private final Logger logger;
        private StreamPredictActionListener<MLTaskResponse, ?> streamActionListener;
        private final String llmInterface;
        private volatile AtomicBoolean isStreamClosed;

        public AwsEventSourceListener(
            final Logger logger,
            StreamPredictActionListener<MLTaskResponse, ?> streamActionListener,
            String llmInterface
        ) {
            this.logger = logger;
            this.streamActionListener = streamActionListener;
            this.llmInterface = llmInterface;
            this.isStreamClosed = new AtomicBoolean(false);
        }

        /***
         * Callback when the SSE endpoint connection is made.
         * @param eventSource the event source
         * @param response the response
         */
        @Override
        public void onOpen(EventSource eventSource, Response response) {
            logger.debug("Connected to SSE Endpoint.");
        }

        /***
         * For each event received from the SSE endpoint
         * @param eventSource The event source
         * @param id The id of the event
         * @param type The type of the event which is used to filter
         * @param data The event data
         */
        @Override
        public void onEvent(EventSource eventSource, String id, String type, String data) {
            log.debug("The event id is {} and the type is {}.", id, type);
            log.debug("The data is: {}", data);
            switch (llmInterface) {
                case LLM_INTERFACE_BEDROCK_CONVERSE_CLAUDE:
                    onClaudeEvent(data);
                    break;
                default:
                    throw new MLException(String.format("Unsupported llm interface: %s", llmInterface));
            }
        }

        /***
         * When the connection is closed we receive this even which is currently only logged.
         * @param eventSource The event source
         */
        @Override
        public void onClosed(EventSource eventSource) {
            logger.debug("SSE CLOSED.");
        }

        /***
         * If there is any failure we log the error and the stack trace
         * During stream resets with no errors we set the connected flag to false to allow the main thread to attempt a re-connect
         * @param eventSource The event source
         * @param t The error object
         * @param response The response
         */
        @Override
        public void onFailure(EventSource eventSource, Throwable t, Response response) {
            logger.error("SSE failure.");
            if (t != null) {
                logger.error("Error: " + t.getMessage(), t);
                if (t instanceof StreamResetException && t.getMessage().contains("NO_ERROR")) {
                    // TODO: reconnect
                } else {
                    streamActionListener.onFailure(new MLException("SSE failure.", t));
                }
            }
        }

        private void onClaudeEvent(String data) {
            Map<String, Object> dataMap = StringUtils.fromJson(data, "data");
            if (dataMap.containsKey("type") && ((String) dataMap.get("type")).contentEquals("message_stop")) {
                sendCompletionResponse();
                return;
            }
            String deltaContent = JsonPath.read(dataMap, "$.output[0].message.content");
            log.info("deltaContent {}", deltaContent);
            if (deltaContent != null && !deltaContent.isEmpty()) {
                log.info("Streaming content: {}", deltaContent);
                sendContentResponse(deltaContent, false);
            }
        }

        private void sendContentResponse(String content, boolean isLast) {
            log.info("sendContentResponse called with content: '{}', isLast: {}", content, isLast);
            List<ModelTensor> modelTensors = new ArrayList<>();
            Map<String, Object> dataMap = Map.of("content", content, "is_last", isLast);

            modelTensors.add(ModelTensor.builder().name("response").dataAsMap(dataMap).build());
            ModelTensorOutput output = ModelTensorOutput
                .builder()
                .mlModelOutputs(List.of(ModelTensors.builder().mlModelTensors(modelTensors).build()))
                .build();
            MLTaskResponse response = MLTaskResponse.builder().output(output).build();
            log.info("Calling streamActionListener.onStreamResponse with isLast: {}", isLast);
            log.info("streamActionListener class: {}", streamActionListener.getClass().getName());
            streamActionListener.onStreamResponse(response, isLast);
            log.info("streamActionListener.onStreamResponse completed for isLast: {}", isLast);
        }

        private void sendCompletionResponse() {
            log.info("Sending completion response");
            if (isStreamClosed.compareAndSet(false, true)) {
                sendContentResponse("", true);
            }
        }
    }
}
