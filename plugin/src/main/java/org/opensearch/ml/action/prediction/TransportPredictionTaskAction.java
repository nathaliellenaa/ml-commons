/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.prediction;

import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_MODEL_AUTO_DEPLOY_ENABLE;
import static org.opensearch.ml.utils.MLExceptionUtils.LOCAL_MODEL_DISABLED_ERR_MSG;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.transport.TransportResponse;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.helper.ModelAccessControlHelper;
import org.opensearch.ml.model.MLModelCacheHelper;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.task.MLPredictTaskRunner;
import org.opensearch.ml.task.MLTaskRunner;
import org.opensearch.ml.utils.MLNodeUtils;
import org.opensearch.ml.utils.RestActionUtils;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.remote.metadata.client.SdkClient;
import org.opensearch.tasks.Task;
import org.opensearch.transport.StreamTransportService;
import org.opensearch.transport.TransportRequest;
import org.opensearch.transport.TransportService;
import org.opensearch.transport.client.Client;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import lombok.extern.log4j.Log4j2;

@Log4j2
@FieldDefaults(level = AccessLevel.PRIVATE)
public class TransportPredictionTaskAction extends HandledTransportAction<ActionRequest, MLTaskResponse> {
    MLTaskRunner<MLPredictionTaskRequest, MLTaskResponse> mlPredictTaskRunner;
    TransportService transportService;
    MLModelCacheHelper modelCacheHelper;

    Client client;
    SdkClient sdkClient;

    ClusterService clusterService;

    NamedXContentRegistry xContentRegistry;

    MLModelManager mlModelManager;

    ModelAccessControlHelper modelAccessControlHelper;

    private volatile boolean enableAutomaticDeployment;

    private MLFeatureEnabledSetting mlFeatureEnabledSetting;

    private StreamTransportService streamTransportService;

    private ActionFilters actionFilters; // Store for debugging

    public class StreamPredictActionListener<Response extends TransportResponse, Request extends TransportRequest>
        implements
            ActionListener<Response> {

        // Remove the TransportChannel field
        // private final TransportChannel channel;

        private final Request request;
        private final String actionName;
        private final ActionListener<Response> delegateListener;

        // Change the constructor signature
        public StreamPredictActionListener(ActionListener<Response> delegateListener, String actionName, Request request) {
            this.delegateListener = delegateListener; // Store the delegate listener
            this.request = request;
            this.actionName = actionName;
        }

        /**
         * Send streaming responses
         * This allows multiple responses to be sent for a single request.
         *
         * @param response    the intermediate response to send
         * @param isLastBatch whether this response is the last one
         */
        public void onStreamResponse(Response response, boolean isLastBatch) {
            log.info("StreamPredictActionListener received response, isLastBatch: {}", isLastBatch);

            // Forward ALL responses to the delegate listener
            delegateListener.onResponse(response);
        }

        /**
         * Reuse ActionListener method to send the last stream response
         * This maintains compatibility on data node side
         *
         * @param response the response to send
         */
        @Override
        public final void onResponse(Response response) {
            onStreamResponse(response, true);
        }

        @Override
        public void onFailure(Exception e) {
            delegateListener.onFailure(e);
        }
    }

    @Inject
    public TransportPredictionTaskAction(
        TransportService transportService,
        ActionFilters actionFilters,
        MLModelCacheHelper modelCacheHelper,
        MLPredictTaskRunner mlPredictTaskRunner,
        ClusterService clusterService,
        Client client,
        SdkClient sdkClient,
        NamedXContentRegistry xContentRegistry,
        MLModelManager mlModelManager,
        ModelAccessControlHelper modelAccessControlHelper,
        MLFeatureEnabledSetting mlFeatureEnabledSetting,
        Settings settings,
        StreamTransportService streamTransportService
    ) {
        super(MLPredictionTaskAction.NAME, transportService, actionFilters, MLPredictionTaskRequest::new);
        this.actionFilters = actionFilters; // Store for debugging
        this.mlPredictTaskRunner = mlPredictTaskRunner;
        this.transportService = transportService;
        this.modelCacheHelper = modelCacheHelper;
        this.clusterService = clusterService;
        this.client = client;
        this.sdkClient = sdkClient;
        this.xContentRegistry = xContentRegistry;
        this.mlModelManager = mlModelManager;
        this.modelAccessControlHelper = modelAccessControlHelper;
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
        this.streamTransportService = streamTransportService;
        enableAutomaticDeployment = ML_COMMONS_MODEL_AUTO_DEPLOY_ENABLE.get(settings);
        clusterService
            .getClusterSettings()
            .addSettingsUpdateConsumer(ML_COMMONS_MODEL_AUTO_DEPLOY_ENABLE, it -> enableAutomaticDeployment = it);
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLTaskResponse> listener) {
        MLPredictionTaskRequest mlPredictionTaskRequest = MLPredictionTaskRequest.fromActionRequest(request);
        log.info("Starting doExecute");

        ThreadContext threadContext = client.threadPool().getThreadContext();
        log.info("All headers BEFORE test on {}: {}", clusterService.localNode().getName(), threadContext.getHeaders());

        log.info("ThreadPool on {}: {}", clusterService.localNode().getName(), client.threadPool().hashCode());
        log.info("ThreadContext on {}: {}", clusterService.localNode().getName(), threadContext.hashCode());
        // log.info("persistent requestId: {}", threadContext.getHeader("_opensearch_ml_streaming_request_id"));
        log.info("persistent requestId: {}", threadContext.getHeader(Task.X_OPAQUE_ID));
        Object requestIdObj = threadContext.getHeader(Task.X_OPAQUE_ID);
        String requestId = requestIdObj != null ? requestIdObj.toString() : null;

        // Check if this is a streaming request
        final boolean isStreamingRequest = true;
        // MLInput mlInput = mlPredictionTaskRequest.getMlInput();
        // if (mlInput instanceof RemoteInferenceMLInput) {
        // RemoteInferenceInputDataSet inputDataSet = (RemoteInferenceInputDataSet) mlInput.getInputDataset();
        // isStreamingRequest = inputDataSet.getParameters() != null &&
        // "true".equals(inputDataSet.getParameters().get("stream"));
        // } else {
        // isStreamingRequest = false;
        // }
        log.info("StreamRequest is {}", isStreamingRequest);

        String modelId = mlPredictionTaskRequest.getModelId();
        String tenantId = mlPredictionTaskRequest.getTenantId();
        if (!TenantAwareHelper.validateTenantId(mlFeatureEnabledSetting, tenantId, listener)) {
            return;
        }
        User user = mlPredictionTaskRequest.getUser();
        if (user == null) {
            user = RestActionUtils.getUserContext(client);
            mlPredictionTaskRequest.setUser(user);
        }
        final User userInfo = user;

        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            log.info("Inside stash - requestId: {}", threadContext.getPersistent(Task.X_OPAQUE_ID));
            log.info("Inside stash header - requestId: {}", threadContext.getHeader(Task.X_OPAQUE_ID));
            if (requestId != null && threadContext.getPersistent(Task.X_OPAQUE_ID) == null) {
                threadContext.putPersistent(Task.X_OPAQUE_ID, requestId);
            }
            if (requestId != null && threadContext.getHeader(Task.X_OPAQUE_ID) == null) {
                threadContext.putHeader(Task.X_OPAQUE_ID, requestId);
            }
            ActionListener<MLTaskResponse> wrappedListener = ActionListener.runBefore(listener, context::restore);
            MLModel cachedMlModel = modelCacheHelper.getModelInfo(modelId);
            ActionListener<MLModel> modelActionListener = new ActionListener<>() {
                @Override
                public void onResponse(MLModel mlModel) {
                    context.restore();
                    Object requestIdObj = threadContext.getPersistent(Task.X_OPAQUE_ID);
                    String requestId = requestIdObj != null ? requestIdObj.toString() : null;
                    log.info("After restore - requestId: {}", requestId);
                    if (isStreamingRequest) {
                        if (client.threadPool().getThreadContext().getPersistent("ml.streaming.enabled") == null) {
                            client.threadPool().getThreadContext().putPersistent("ml.streaming.enabled", "true");
                        }
                        // client.threadPool().getThreadContext().putHeader("ml.streaming.original.listener", listener);
                        // The REST channel should already be stored by the REST layer
                        log.info("Streaming enabled, REST channel should be available in ThreadContext");
                    }

                    modelCacheHelper.setModelInfo(modelId, mlModel);
                    FunctionName functionName = mlModel.getAlgorithm();
                    if (FunctionName.isDLModel(functionName) && !mlFeatureEnabledSetting.isLocalModelEnabled()) {
                        throw new IllegalStateException(LOCAL_MODEL_DISABLED_ERR_MSG);
                    }
                    mlPredictionTaskRequest.getMlInput().setAlgorithm(functionName);
                    modelAccessControlHelper
                        .validateModelGroupAccess(
                            userInfo,
                            mlFeatureEnabledSetting,
                            tenantId,
                            mlModel.getModelGroupId(),
                            client,
                            sdkClient,
                            ActionListener.wrap(access -> {
                                if (!access) {
                                    wrappedListener
                                        .onFailure(
                                            new OpenSearchStatusException(
                                                "User Doesn't have privilege to perform this operation on this model",
                                                RestStatus.FORBIDDEN
                                            )
                                        );
                                } else {
                                    if (modelCacheHelper.getIsModelEnabled(modelId) != null
                                        && !modelCacheHelper.getIsModelEnabled(modelId)) {
                                        wrappedListener
                                            .onFailure(new OpenSearchStatusException("Model is disabled.", RestStatus.FORBIDDEN));
                                    } else {
                                        if (FunctionName.isDLModel(functionName)) {
                                            if (modelCacheHelper.getRateLimiter(modelId) != null
                                                && !modelCacheHelper.getRateLimiter(modelId).request()) {
                                                wrappedListener
                                                    .onFailure(
                                                        new OpenSearchStatusException(
                                                            "Request is throttled at model level.",
                                                            RestStatus.TOO_MANY_REQUESTS
                                                        )
                                                    );
                                            } else if (userInfo != null
                                                && modelCacheHelper.getUserRateLimiter(modelId, userInfo.getName()) != null
                                                && !modelCacheHelper.getUserRateLimiter(modelId, userInfo.getName()).request()) {
                                                wrappedListener
                                                    .onFailure(
                                                        new OpenSearchStatusException(
                                                            "Request is throttled at user level. If you think there's an issue, please contact your cluster admin.",
                                                            RestStatus.TOO_MANY_REQUESTS
                                                        )
                                                    );
                                            } else {
                                                validateInputSchema(modelId, mlPredictionTaskRequest.getMlInput());
                                                log.info("Stream request here is {}", isStreamingRequest);
                                                if (isStreamingRequest) {
                                                    log.info("Executing streaming prediction for model: {}", modelId);

                                                    // Create streaming listener that forwards all responses
                                                    StreamPredictActionListener<MLTaskResponse, MLPredictionTaskRequest> streamingListener =
                                                        new StreamPredictActionListener<>(
                                                            wrappedListener,
                                                            "ml_predict_stream",
                                                            mlPredictionTaskRequest
                                                        );

                                                    // Store the streaming listener in ThreadContext so the executor can use it
                                                    client
                                                        .threadPool()
                                                        .getThreadContext()
                                                        .putTransient("ml.streaming.listener", streamingListener);

                                                    // Store streaming transport service for cross-node streaming
                                                    if (client
                                                        .threadPool()
                                                        .getThreadContext()
                                                        .getPersistent("ml.streaming.transport.service") == null) {
                                                        client
                                                            .threadPool()
                                                            .getThreadContext()
                                                            .putPersistent("ml.streaming.transport.service", streamTransportService);
                                                        log.info("Stored StreamTransportService for cross-node streaming");
                                                    }

                                                    if (requestIdObj != null) {
                                                        // String requestId = requestIdObj.toString();
                                                        client
                                                            .threadPool()
                                                            .getThreadContext()
                                                            .putHeader("ml.streaming.request.id", requestId);
                                                        log.info("Stored streaming request ID for cross-node access: {}", requestId);
                                                    }

                                                    executePredict(mlPredictionTaskRequest, streamingListener, modelId);
                                                } else {
                                                    executePredict(mlPredictionTaskRequest, wrappedListener, modelId);
                                                }
                                            }
                                        } else {
                                            validateInputSchema(modelId, mlPredictionTaskRequest.getMlInput());
                                            executePredict(mlPredictionTaskRequest, wrappedListener, modelId);
                                        }
                                    }
                                }
                            }, wrappedListener::onFailure)
                        );
                }

                @Override
                public void onFailure(Exception e) {
                    wrappedListener.onFailure(e);
                }
            };

            if (cachedMlModel != null) {
                modelActionListener.onResponse(cachedMlModel);
            } else {
                // For multi-node cluster, the function name is null in cache, so should always get model first.
                mlModelManager.getModel(modelId, tenantId, modelActionListener);
            }
        } catch (Exception e) {
            log.error("Failed to predict " + mlPredictionTaskRequest.toString(), e);
            listener.onFailure(e);
        }
    }

    private ActionListener<MLTaskResponse> findStreamListener(ActionListener<MLTaskResponse> listener) {
        try {
            // Check if current listener is StreamPredictActionListener
            if (listener.getClass().getSimpleName().contains("StreamPredictActionListener")) {
                return listener;
            }

            // Try to get delegate field from wrapper listeners
            java.lang.reflect.Field[] fields = listener.getClass().getDeclaredFields();
            for (java.lang.reflect.Field field : fields) {
                field.setAccessible(true);
                Object value = field.get(listener);
                if (value instanceof ActionListener) {
                    ActionListener<MLTaskResponse> delegateListener = (ActionListener<MLTaskResponse>) value;
                    ActionListener<MLTaskResponse> found = findStreamListener(delegateListener);
                    if (found != null) {
                        return found;
                    }
                }
            }
        } catch (Exception e) {
            log.debug("Error traversing listener chain", e);
        }
        return null;
    }

    private void executeStreamingPredict(
        MLPredictionTaskRequest request,
        StreamPredictActionListener<MLTaskResponse, MLPredictionTaskRequest> streamingListener,
        String modelId
    ) {
        try {
            // Call the task runner with streaming support
            if (mlPredictTaskRunner instanceof MLPredictTaskRunner) {
                log.info("Goes here!");
                // ((MLPredictTaskRunner) mlPredictTaskRunner).runWithStreaming(request, streamingListener);
            } else {
                streamingListener.onFailure(new UnsupportedOperationException("Streaming not supported by this task runner"));
            }
        } catch (Exception e) {
            log.error("Failed to execute streaming prediction", e);
            streamingListener.onFailure(e);
        }
    }

    private void executePredict(
        MLPredictionTaskRequest mlPredictionTaskRequest,
        ActionListener<MLTaskResponse> wrappedListener,
        String modelId
    ) {
        String requestId = mlPredictionTaskRequest.getRequestID();
        log.debug("receive predict request {} for model {}", requestId, mlPredictionTaskRequest.getModelId());

        // Ensure persistent header is set before transport call
        ThreadContext threadContext = client.threadPool().getThreadContext();
        Object persistentRequestIdObj = threadContext.getPersistent(Task.X_OPAQUE_ID);
        String persistentRequestId = persistentRequestIdObj != null ? persistentRequestIdObj.toString() : null;
        log.info("persistentRequestId {}", persistentRequestId);
        if (persistentRequestId != null && threadContext.getHeader(Task.X_OPAQUE_ID) == null) {
            threadContext.putHeader(Task.X_OPAQUE_ID, persistentRequestId);
        }

        long startTime = System.nanoTime();
        // For remote text embedding model, neural search will set mlPredictionTaskRequest.getMlInput().getAlgorithm() as
        // TEXT_EMBEDDING. In ml-commons we should always use the real function name of model: REMOTE. So we try to get
        // from model cache first.
        FunctionName functionName = modelCacheHelper
            .getOptionalFunctionName(modelId)
            .orElse(mlPredictionTaskRequest.getMlInput().getAlgorithm());
        mlPredictTaskRunner
            .run(
                // This is by design to NOT use mlPredictionTaskRequest.getMlInput().getAlgorithm() here
                functionName,
                mlPredictionTaskRequest,
                transportService,
                ActionListener.runAfter(wrappedListener, () -> {
                    long endTime = System.nanoTime();
                    double durationInMs = (endTime - startTime) / 1e6;
                    modelCacheHelper.addPredictRequestDuration(modelId, durationInMs);
                    modelCacheHelper.refreshLastAccessTime(modelId);
                    log.debug("completed predict request {} for model {}", requestId, modelId);
                })
            );
    }

    public void validateInputSchema(String modelId, MLInput mlInput) {
        if (modelCacheHelper.getModelInterface(modelId) != null && modelCacheHelper.getModelInterface(modelId).get("input") != null) {
            String inputSchemaString = modelCacheHelper.getModelInterface(modelId).get("input");
            try {
                String InputString = mlInput.toXContent(XContentFactory.jsonBuilder(), ToXContent.EMPTY_PARAMS).toString();
                // Process the parameters field in the input dataset to convert it back to its original datatype, instead of a string
                String processedInputString = MLNodeUtils.processRemoteInferenceInputDataSetParametersValue(InputString, inputSchemaString);
                MLNodeUtils.validateSchema(inputSchemaString, processedInputString);
            } catch (Exception e) {
                throw new OpenSearchStatusException(
                    "Error validating input schema, if you think this is expected, please update your 'input' field in the 'interface' field for this model: "
                        + e.getMessage(),
                    RestStatus.BAD_REQUEST
                );
            }
        }
    }

}
