/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.utils.MLExceptionUtils.BATCH_INFERENCE_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.MLExceptionUtils.LOCAL_MODEL_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.MLExceptionUtils.REMOTE_INFERENCE_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.MLExceptionUtils.STREAM_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_ALGORITHM;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_MODEL_ID;
import static org.opensearch.ml.utils.RestActionUtils.getActionTypeFromRestRequest;
import static org.opensearch.ml.utils.RestActionUtils.getParameterId;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.arrow.spi.StreamTicket;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.lease.Releasable;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.support.XContentHttpChunk;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.transport.TransportResponse;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.http.HttpChunk;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.connector.ConnectorAction.ActionType;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.remote.RemoteInferenceMLInput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.engine.algorithms.remote.StreamingRegistry;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.plugin.MachineLearningPlugin;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.StreamingRestChannel;
import org.opensearch.tasks.Task;
import org.opensearch.transport.StreamTransportService;
import org.opensearch.transport.TransportRequest;
import org.opensearch.transport.client.node.NodeClient;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;

import lombok.extern.log4j.Log4j2;
import reactor.core.publisher.Flux;

@Log4j2
public class RestMLPredictionStreamingAction extends BaseRestHandler {

    public class StreamPredictActionListener<Response extends TransportResponse, Request extends TransportRequest>
        implements
            ActionListener<Response> {

        private final StreamingRestChannel restChannel;
        private final String actionName;
        private final Request request;
        private int chunkCount = 0;
        private final NodeClient client;

        // Constructor for REST layer
        public StreamPredictActionListener(StreamingRestChannel restChannel, String actionName, Request request, NodeClient client) {
            this.restChannel = restChannel;
            this.actionName = actionName;
            this.request = request;
            this.client = client;
        }

        // Add getter method
        public StreamingRestChannel getRestChannel() {
            return restChannel;
        }

        public void onStreamResponse(Response response, boolean isLastBatch) {
            log.info("REST onStreamResponse received, isLastBatch: {}", isLastBatch);

            MLTaskResponse mlResponse = (MLTaskResponse) response;
            String content = extractContent(mlResponse);
            boolean isLast = isLastChunk(mlResponse);
            log.info("Extracted content: '{}', isLast from content: {}", content, isLast);

            // Always send chunks to see what's happening
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
            // Store the StreamingRestChannel in ThreadContext for cross-node access
            if (client.threadPool().getThreadContext().getPersistent("ml.streaming.rest.channel") == null) {
                client.threadPool().getThreadContext().putPersistent("ml.streaming.rest.channel", restChannel);
            }

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

    // Add this at the end of your class
    public static class MLStreamRequest extends ActionRequest {
        private final byte[] ticketData;

        public MLStreamRequest(StreamTicket ticket) {
            // Convert ticket to byte array for serialization
            try {
                BytesStreamOutput out = new BytesStreamOutput();
                // Assuming StreamTicket has some serializable data
                out.writeString(ticket.toString());
                this.ticketData = out.bytes().toBytesRef().bytes;
            } catch (IOException e) {
                throw new RuntimeException("Failed to serialize ticket", e);
            }
        }

        public MLStreamRequest(StreamInput in) throws IOException {
            super(in);
            this.ticketData = in.readByteArray();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeByteArray(ticketData);
        }

        @Override
        public ActionRequestValidationException validate() {
            return null;
        }

        public byte[] getTicketData() {
            return ticketData;
        }
    }

    private static final String ML_PREDICTION_ACTION = "ml_prediction_streaming_action";

    private MLModelManager modelManager;

    private MLFeatureEnabledSetting mlFeatureEnabledSetting;

    private MachineLearningPlugin.StreamManagerWrapper streamManagerWrapper;

    private StreamTransportService streamTransportService;

    private ClusterService clusterService;

    /**
     * Constructor
     */
    public RestMLPredictionStreamingAction(
        MLModelManager modelManager,
        MLFeatureEnabledSetting mlFeatureEnabledSetting,
        MachineLearningPlugin.StreamManagerWrapper streamManagerWrapper,
        StreamTransportService streamTransportService,
        ClusterService clusterService
    ) {
        this.modelManager = modelManager;
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
        this.streamManagerWrapper = streamManagerWrapper;
        this.streamTransportService = streamTransportService;
        this.clusterService = clusterService;
    }

    @Override
    public String getName() {
        return ML_PREDICTION_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
            .of(
                new Route(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/{%s}/_predict/stream", ML_BASE_URI, PARAMETER_MODEL_ID)
                ),
                new Route(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/{%s}/_batch_predict/stream", ML_BASE_URI, PARAMETER_MODEL_ID)
                )
            );
    }

    @Override
    public RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        if (!mlFeatureEnabledSetting.isStreamEnabled()) {
            throw new IllegalStateException(STREAM_DISABLED_ERR_MSG);
        }
        String algorithm = request.param(PARAMETER_ALGORITHM);
        String modelId = getParameterId(request, PARAMETER_MODEL_ID);
        Optional<FunctionName> functionName = modelManager.getOptionalModelFunctionName(modelId);

        if (algorithm == null && functionName.isPresent()) {
            algorithm = functionName.get().name();
        }

        final StreamingRestChannelConsumer consumer = (channel) -> {
            final MediaType mediaType = request.getMediaType();
            channel.prepareResponse(RestStatus.OK, Map.of("Content-Type", List.of(mediaType.mediaTypeWithoutParameters())));
            // Flux
            // .from(channel)
            // .ofType(HttpChunk.class)
            // .takeUntil(HttpChunk::isLast)
            // .map(HttpChunk::content)
            // .reduce(CompositeBytesReference::of)
            // .doOnSuccess(bytesReference -> {
            Flux
                .from(channel)
                .ofType(HttpChunk.class)
                .take(1)  // Take only the first chunk (request body)
                .map(HttpChunk::content)
                .doOnNext(bytesReference -> {  // Execute immediately for each chunk
                    try {
                        MLPredictionTaskRequest taskRequest = getRequest(modelId, FunctionName.REMOTE.name(), request, bytesReference);
                        ThreadContext threadContext = client.threadPool().getThreadContext();
                        String requestId = UUID.randomUUID().toString();
                        StreamingRegistry.register(requestId, channel);
                        if (threadContext.getPersistent(Task.X_OPAQUE_ID) == null) {
                            threadContext.putPersistent(Task.X_OPAQUE_ID, requestId);
                        }
                        if (threadContext.getPersistent("ml.streaming.rest.channel") == null) {
                            threadContext.putPersistent("ml.streaming.rest.channel", channel);
                        }
                        // Add debug logging here - before transport
                        log.info("Before transport - headers: {}", threadContext.getHeaders());
                        log.info("Before transport - persistent testHeader: {}", threadContext.getPersistent(Task.X_OPAQUE_ID));
                        log.info("Before transport - requestId: {}", requestId);
                        log.info("Just before client.execute - persistent header: {}", threadContext.getPersistent(Task.X_OPAQUE_ID));
                        log
                            .info(
                                "Just before client.execute - persistent channel: {}",
                                threadContext.getPersistent("ml.streaming.rest.channel")
                            );

                        // client.threadPool().getThreadContext().putTransient("ml.streaming.rest.channel", channel);
                        client
                            .execute(
                                MLPredictionTaskAction.INSTANCE,
                                taskRequest,
                                new StreamPredictActionListener<MLTaskResponse, MLPredictionTaskRequest>(
                                    channel,           // StreamingRestChannel
                                    "ml_predict_stream", // String actionName
                                    taskRequest,       // MLPredictionTaskRequest
                                    client
                                )
                            );
                    } catch (IOException e) {
                        throw new MLException("Got an exception in flux.", e);
                    }
                })
                .onErrorComplete(ex -> {
                    if (ex instanceof Error) {
                        log.info("Got an error in flux");
                        return false;
                    }
                    try {
                        channel.sendResponse(new BytesRestResponse(channel, (Exception) ex));
                        return true;
                    } catch (final IOException e) {
                        throw new UncheckedIOException(e);
                    }
                })
                .subscribe();
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

    /**
     * Creates a MLPredictionTaskRequest from a RestRequest
     *
     * @param request RestRequest
     * @return MLPredictionTaskRequest
     */
    @VisibleForTesting
    MLPredictionTaskRequest getRequest(String modelId, String algorithm, RestRequest request, BytesReference content) throws IOException {
        ActionType actionType = ActionType.from(getActionTypeFromRestRequest(request));
        if (FunctionName.REMOTE.name().equals(algorithm) && !mlFeatureEnabledSetting.isRemoteInferenceEnabled()) {
            throw new IllegalStateException(REMOTE_INFERENCE_DISABLED_ERR_MSG);
        } else if (FunctionName.isDLModel(FunctionName.from(algorithm.toUpperCase(Locale.ROOT)))
            && !mlFeatureEnabledSetting.isLocalModelEnabled()) {
            throw new IllegalStateException(LOCAL_MODEL_DISABLED_ERR_MSG);
        } else if (ActionType.BATCH_PREDICT == actionType && !mlFeatureEnabledSetting.isOfflineBatchInferenceEnabled()) {
            throw new IllegalStateException(BATCH_INFERENCE_DISABLED_ERR_MSG);
        } else if (!ActionType.isValidActionInModelPrediction(actionType)) {
            throw new IllegalArgumentException("Wrong action type in the rest request path!");
        }

        XContentParser parser = request
            .getMediaType()
            .xContent()
            .createParser(request.getXContentRegistry(), LoggingDeprecationHandler.INSTANCE, content.streamInput());

        // XContentParser parser = request.contentParser();
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
        MLInput mlInput = MLInput.parse(parser, algorithm, actionType);
        if (FunctionName.REMOTE.name().contentEquals(algorithm)) {
            RemoteInferenceMLInput input = (RemoteInferenceMLInput) mlInput;
            RemoteInferenceInputDataSet inputDataSet = (RemoteInferenceInputDataSet) input.getInputDataset();
            inputDataSet.getParameters().put("stream", String.valueOf(true));
            return new MLPredictionTaskRequest(modelId, input, null, null);
        }
        return new MLPredictionTaskRequest(modelId, mlInput, null, null);
    }

    private HttpChunk createHttpChunkFromEvent(byte[] event) {
        BytesReference content = BytesReference.fromByteBuffer(ByteBuffer.wrap(event));
        return new HttpChunk() {
            @Override
            public void close() {
                if (content instanceof Releasable) {
                    ((Releasable) content).close();
                }
            }

            @Override
            public boolean isLast() {
                return false;
            }

            @Override
            public BytesReference content() {
                return content;
            }
        };
    }

    private HttpChunk convertToHttpChunk(MLTaskResponse response) throws IOException {
        String content = "";
        boolean isLast = false;

        // Extract content and is_last flag
        try {
            ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
            if (output != null && !output.getMlModelOutputs().isEmpty()) {
                ModelTensors modelTensors = output.getMlModelOutputs().get(0);
                if (!modelTensors.getMlModelTensors().isEmpty()) {
                    Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(0).getDataAsMap();
                    if (dataMap.containsKey("content")) {
                        content = (String) dataMap.get("content");
                        if (content == null) {
                            content = "";
                        }
                    }
                    if (dataMap.containsKey("is_last")) {
                        isLast = Boolean.TRUE.equals(dataMap.get("is_last"));
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to extract content from response", e);
            content = "";
        }

        log.info("Converting to HttpChunk - content: '{}', isLast: {}", content, isLast);

        // Create JSON response with both content and debug info
        Map<String, Object> responseMap = Map.of("content", content, "is_last", isLast);
        XContentBuilder builder = XContentFactory.jsonBuilder().map(responseMap);
        BytesReference bytesRef = BytesReference.bytes(builder);

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
}
