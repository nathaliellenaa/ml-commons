/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.plugin.MachineLearningPlugin.STREAM_EXECUTE_THREAD_POOL;
import static org.opensearch.ml.utils.MLExceptionUtils.AGENT_FRAMEWORK_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.MLExceptionUtils.STREAM_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_AGENT_ID;
import static org.opensearch.ml.utils.RestActionUtils.getAlgorithm;
import static org.opensearch.ml.utils.RestActionUtils.isAsync;
import static org.opensearch.ml.utils.TenantAwareHelper.getTenantID;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.lease.Releasable;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.support.XContentHttpChunk;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.transport.TransportResponse;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.http.HttpChunk;
import org.opensearch.ml.action.execute.TransportExecuteStreamingTaskAction;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.Input;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.execute.agent.AgentMLInput;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.execute.MLExecuteStreamingTaskAction;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskRequest;
import org.opensearch.ml.repackage.com.google.common.annotations.VisibleForTesting;
import org.opensearch.ml.repackage.com.google.common.collect.ImmutableList;
import org.opensearch.ml.utils.error.ErrorMessage;
import org.opensearch.ml.utils.error.ErrorMessageFactory;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.StreamingRestChannel;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.StreamTransportResponseHandler;
import org.opensearch.transport.TransportException;
import org.opensearch.transport.TransportRequest;
import org.opensearch.transport.TransportRequestOptions;
import org.opensearch.transport.client.node.NodeClient;
import org.opensearch.transport.stream.StreamTransportResponse;

import lombok.extern.log4j.Log4j2;
import reactor.core.publisher.Flux;

@Log4j2
public class RestMLExecuteStreamingAction extends BaseRestHandler {

    public class StreamExecuteActionListener<Response extends TransportResponse, Request extends TransportRequest>
        implements
            ActionListener<Response> {

        private final StreamingRestChannel restChannel;

        // Constructor for REST layer
        public StreamExecuteActionListener(StreamingRestChannel restChannel) {
            this.restChannel = restChannel;
        }

        public void onStreamResponse(Response response, boolean isLastBatch) {
            log.info("REST onStreamResponse received, isLastBatch: {}", isLastBatch);

            MLTaskResponse mlResponse = (MLTaskResponse) response;
            String content = extractContent(mlResponse);
            boolean isLast = isLastChunk(mlResponse);
            log.info("Extracted content: '{}', isLast from content: {}", content, isLast);

            try {
                restChannel.sendChunk(convertToHttpChunk(mlResponse));
                log.info("Sent chunk with content: '{}', isLast: {}", content, isLast);
            } catch (IOException e) {
                log.error("Failed to send chunk", e);
            }

            // Send final marker when stream is complete
            if (isLast) {
                log.info("Stream completed - sending final marker");
                restChannel.sendChunk(XContentHttpChunk.last());
            }
        }

        private String extractContent(MLTaskResponse response) {
            try {
                ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
                if (output != null && !output.getMlModelOutputs().isEmpty()) {
                    ModelTensors modelTensors = output.getMlModelOutputs().get(0);
                    if (!modelTensors.getMlModelTensors().isEmpty()) {
                        Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(0).getDataAsMap();
                        if (dataMap.containsKey("content")) {
                            return (String) dataMap.get("content");
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Failed to extract content", e);
            }
            return "";
        }

        private boolean isLastChunk(MLTaskResponse response) {
            try {
                ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
                if (output != null && !output.getMlModelOutputs().isEmpty()) {
                    ModelTensors modelTensors = output.getMlModelOutputs().get(0);
                    if (!modelTensors.getMlModelTensors().isEmpty()) {
                        Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(0).getDataAsMap();
                        if (dataMap.containsKey("is_last")) {
                            return Boolean.TRUE.equals(dataMap.get("is_last"));
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Failed to check is_last", e);
            }
            return false;
        }

        @Override
        public final void onResponse(Response response) {
            MLTaskResponse mlResponse = (MLTaskResponse) response;
            boolean isLastBatch = isLastChunk(mlResponse);
            log.info("islastBatch from content is {}", isLastBatch);
            onStreamResponse(response, isLastBatch);
        }

        @Override
        public void onFailure(Exception e) {
            throw new MLException("Got an exception in MLPredictionTaskAction.", e);
        }
    }

    private static final String ML_EXECUTE_ACTION = "ml_execute_streaming_action";
    private final MLFeatureEnabledSetting mlFeatureEnabledSetting;
    private ClusterService clusterService;

    /**
     * Constructor
     */
    public RestMLExecuteStreamingAction(MLFeatureEnabledSetting mlFeatureEnabledSetting, ClusterService clusterService) {
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
        this.clusterService = clusterService;
    }

    @Override
    public String getName() {
        return ML_EXECUTE_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
            .of(
                new Route(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/agents/{%s}/_execute/stream", ML_BASE_URI, PARAMETER_AGENT_ID)
                )
            );
    }

    @Override
    public RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        if (!mlFeatureEnabledSetting.isStreamEnabled()) {
            throw new IllegalStateException(STREAM_DISABLED_ERR_MSG);
        }

        String agentId = request.param(PARAMETER_AGENT_ID);

        final StreamingRestChannelConsumer consumer = (channel) -> {
            Map<String, List<String>> headers = Map
                .of(
                    "Content-Type",
                    List.of("text/event-stream"),
                    "Cache-Control",
                    List.of("no-cache"),
                    "Connection",
                    List.of("keep-alive")
                );
            channel.prepareResponse(RestStatus.OK, headers);

            Flux.from(channel).ofType(HttpChunk.class).take(1).map(HttpChunk::content).doOnNext(bytesReference -> {
                try {
                    MLExecuteTaskRequest mlExecuteTaskRequest = getRequest(agentId, request, bytesReference);
                    StreamTransportResponseHandler<MLTaskResponse> handler = new StreamTransportResponseHandler<MLTaskResponse>() {
                        @Override
                        public void handleStreamResponse(StreamTransportResponse<MLTaskResponse> streamResponse) {
                            try {
                                // Process one response at a time
                                MLTaskResponse response = streamResponse.nextResponse();

                                if (response != null) {
                                    log.info("Received response: {}", response);
                                    boolean isLast = isLastChunk(response);

                                    HttpChunk chunk = convertToHttpChunk(response);
                                    channel.sendChunk(chunk);

                                    if (isLast) {
                                        log.info("Detected is_last=true, completing stream");
                                        channel.sendChunk(XContentHttpChunk.last());
                                        streamResponse.close();
                                        return;
                                    }

                                    // Recursively handle the next response - asynchronously
                                    client
                                        .threadPool()
                                        .executor(STREAM_EXECUTE_THREAD_POOL)
                                        .execute(() -> handleStreamResponse(streamResponse));
                                } else {
                                    log.info("No more responses, closing stream");
                                    channel.sendChunk(XContentHttpChunk.last());
                                    streamResponse.close();
                                }
                            } catch (Exception e) {
                                streamResponse.cancel("Error processing stream", e);
                                log.error("Error in stream handling", e);
                            }
                        }

                        @Override
                        public void handleException(TransportException exp) {
                            // fail("Transport exception: " + exp.getMessage());
                        }

                        @Override
                        public String executor() {
                            return ThreadPool.Names.SAME;
                        }

                        @Override
                        public MLTaskResponse read(StreamInput in) throws IOException {
                            return new MLTaskResponse(in);
                        }
                    };

                    TransportExecuteStreamingTaskAction.streamTransportService
                        .sendRequest(
                            clusterService.localNode(),
                            MLExecuteStreamingTaskAction.NAME,
                            mlExecuteTaskRequest,
                            TransportRequestOptions.builder().withType(TransportRequestOptions.Type.STREAM).build(),
                            handler
                        );

                } catch (IOException e) {
                    throw new MLException("Got an exception in flux.", e);
                }
            }).onErrorComplete(ex -> {
                if (ex instanceof Error) {
                    log.error("Got an error in flux");
                    return false;
                }
                try {
                    channel.sendResponse(new BytesRestResponse(channel, (Exception) ex));
                    return true;
                } catch (final IOException e) {
                    throw new UncheckedIOException(e);
                }
            }).subscribe();
        };

        return channel -> {
            if (channel instanceof StreamingRestChannel) {
                consumer.accept((StreamingRestChannel) channel);
            } else {
                final ActionRequestValidationException validationError = new ActionRequestValidationException();
                validationError.addValidationError("Unable to initiate request / response streaming over non-streaming channel");
                channel.sendResponse(new BytesRestResponse(channel, validationError));
            }
        };
    }

    private boolean isLastChunk(MLTaskResponse response) {
        try {
            ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
            if (output != null && !output.getMlModelOutputs().isEmpty()) {
                ModelTensors modelTensors = output.getMlModelOutputs().get(0);
                for (ModelTensor tensor : modelTensors.getMlModelTensors()) {
                    String name = tensor.getName();
                    if (("llm_response".equals(name) || "response".equals(name)) && tensor.getDataAsMap() != null) {
                        Map<String, ?> dataMap = tensor.getDataAsMap();
                        if (dataMap.containsKey("is_last")) {
                            return Boolean.TRUE.equals(dataMap.get("is_last"));
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to check is_last", e);
        }
        return false;
    }


    /**
     * Creates a MLExecuteTaskRequest from a RestRequest
     *
     * @param request RestRequest
     * @return MLExecuteTaskRequest
     */
    @VisibleForTesting
    MLExecuteTaskRequest getRequest(String agentId, RestRequest request, BytesReference content) throws IOException {
        XContentParser parser = request
            .getMediaType()
            .xContent()
            .createParser(request.getXContentRegistry(), LoggingDeprecationHandler.INSTANCE, content.streamInput());
        boolean async = isAsync(request);
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);

        String uri = request.getHttpRequest().uri();
        FunctionName functionName = null;
        Input input = null;
        if (uri.startsWith(ML_BASE_URI + "/agents/")) {
            if (!mlFeatureEnabledSetting.isAgentFrameworkEnabled()) {
                throw new IllegalStateException(AGENT_FRAMEWORK_DISABLED_ERR_MSG);
            }
            String tenantId = getTenantID(mlFeatureEnabledSetting.isMultiTenancyEnabled(), request);
            functionName = FunctionName.AGENT;
            input = MLInput.parse(parser, functionName.name());
            ((AgentMLInput) input).setAgentId(agentId);
            ((AgentMLInput) input).setTenantId(tenantId);
            ((AgentMLInput) input).setIsAsync(async);
        } else {
            String algorithm = getAlgorithm(request).toUpperCase(Locale.ROOT);
            functionName = FunctionName.from(algorithm);
            input = parser.namedObject(Input.class, functionName.name(), null);
        }
        return new MLExecuteTaskRequest(functionName, input);
    }

    @Override
    public boolean supportsContentStream() {
        return true;
    }

    @Override
    public boolean supportsStreaming() {
        return true;
    }

    @Override
    public boolean allowsUnsafeBuffers() {
        return true;
    }

    // private HttpChunk convertToHttpChunk(MLTaskResponse response) throws IOException {
    // String content = "";
    // boolean isLast = false;
    //
    // // Extract content and is_last flag
    // try {
    // ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
    // if (output != null && !output.getMlModelOutputs().isEmpty()) {
    // ModelTensors modelTensors = output.getMlModelOutputs().get(0);
    // if (!modelTensors.getMlModelTensors().isEmpty()) {
    // Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(0).getDataAsMap();
    // if (dataMap.containsKey("content")) {
    // content = (String) dataMap.get("content");
    // if (content == null) {
    // content = "";
    // }
    // }
    // if (dataMap.containsKey("is_last")) {
    // isLast = Boolean.TRUE.equals(dataMap.get("is_last"));
    // }
    // }
    // }
    // } catch (Exception e) {
    // log.error("Failed to extract content from response", e);
    // content = "";
    // }
    //
    // log.info("Converting to HttpChunk - content: '{}', isLast: {}", content, isLast);
    //
    // // Create proper SSE formatted response
    // String jsonData = "{\"content\":\"" + content.replace("\"", "\\\"") + "\",\"is_last\":" + isLast + "}";
    // String sseData = "data: " + jsonData + "\n\n";
    // BytesReference bytesRef = BytesReference.fromByteBuffer(ByteBuffer.wrap(sseData.getBytes()));
    //
    // return new HttpChunk() {
    // @Override
    // public void close() {
    // if (bytesRef instanceof Releasable) {
    // ((Releasable) bytesRef).close();
    // }
    // }
    //
    // @Override
    // public boolean isLast() {
    // return false;
    // }
    //
    // @Override
    // public BytesReference content() {
    // return bytesRef;
    // }
    // };
    // }

    private HttpChunk convertToHttpChunk(MLTaskResponse response) throws IOException {
        String memoryId = "";
        String parentInteractionId = "";
        String content = "";
        boolean isLast = false;

        // Extract values from multiple tensors
        try {
            ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
            if (output != null && !output.getMlModelOutputs().isEmpty()) {
                ModelTensors modelTensors = output.getMlModelOutputs().get(0);
                List<ModelTensor> tensors = modelTensors.getMlModelTensors();

                for (ModelTensor tensor : tensors) {
                    String name = tensor.getName();
                    if ("memory_id".equals(name) && tensor.getResult() != null) {
                        memoryId = tensor.getResult();
                    } else if ("parent_interaction_id".equals(name) && tensor.getResult() != null) {
                        parentInteractionId = tensor.getResult();
                    } else if (("llm_response".equals(name) || "response".equals(name)) && tensor.getDataAsMap() != null) {
                        Map<String, ?> dataMap = tensor.getDataAsMap();
                        if (dataMap.containsKey("content")) {
                            content = (String) dataMap.get("content");
                            if (content == null)
                                content = "";
                        }
                        if (dataMap.containsKey("is_last")) {
                            isLast = Boolean.TRUE.equals(dataMap.get("is_last"));
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to extract values from response", e);
        }

        String finalContent = content;
        boolean finalIsLast = isLast;

        log
            .info(
                "Converting to HttpChunk - memoryId: '{}', parentId: '{}', content: '{}', isLast: {}",
                memoryId,
                parentInteractionId,
                content,
                isLast
            );

        // Create ordered tensors
        List<ModelTensor> orderedTensors = List
            .of(
                ModelTensor.builder().name("memory_id").result(memoryId).build(),
                ModelTensor.builder().name("parent_interaction_id").result(parentInteractionId).build(),
                ModelTensor.builder().name("response").dataAsMap(new LinkedHashMap<String, Object>() {
                    {
                        put("content", finalContent);
                        put("is_last", finalIsLast);
                    }
                }).build()
            );

        ModelTensors tensors = ModelTensors.builder().mlModelTensors(orderedTensors).build();

        ModelTensorOutput tensorOutput = ModelTensorOutput.builder().mlModelOutputs(List.of(tensors)).build();

        XContentBuilder builder = XContentFactory.jsonBuilder();
        tensorOutput.toXContent(builder, ToXContent.EMPTY_PARAMS);
        String jsonData = builder.toString();

        String sseData = "data: " + jsonData + "\n\n";
        BytesReference bytesRef = BytesReference.fromByteBuffer(ByteBuffer.wrap(sseData.getBytes()));

        return new HttpChunk() {
            @Override
            public void close() {
                if (bytesRef instanceof Releasable) {
                    ((Releasable) bytesRef).close();
                }
            }

            @Override
            public boolean isLast() {
                return false;
            }

            @Override
            public BytesReference content() {
                return bytesRef;
            }
        };
    }

    private void reportError(final RestChannel channel, final Exception e, final RestStatus status) {
        ErrorMessage errorMessage = ErrorMessageFactory.createErrorMessage(e, status.getStatus());
        try {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("status", errorMessage.getStatus());
            builder.startObject("error");
            builder.field("type", errorMessage.getType());
            builder.field("reason", errorMessage.getReason());
            builder.field("details", errorMessage.getDetails());
            builder.endObject();
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.fromCode(errorMessage.getStatus()), builder));
        } catch (Exception exception) {
            log.error("Failed to build xContent for an error response, so reply with a plain string.", exception);
            channel.sendResponse(new BytesRestResponse(RestStatus.fromCode(errorMessage.getStatus()), errorMessage.toString()));
        }
    }

    private boolean isClientError(Exception e) {
        return e instanceof IllegalArgumentException || e instanceof IllegalAccessException;
    }
}
