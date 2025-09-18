/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.execute;

import static org.opensearch.ml.plugin.MachineLearningPlugin.STREAM_PREDICT_THREAD_POOL;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.transport.TransportResponse;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.transport.execute.MLExecuteStreamingTaskAction;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskRequest;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskResponse;
import org.opensearch.ml.task.MLExecuteTaskRunner;
import org.opensearch.ml.task.MLTaskRunner;
import org.opensearch.tasks.Task;
import org.opensearch.transport.StreamTransportService;
import org.opensearch.transport.TransportChannel;
import org.opensearch.transport.TransportRequest;
import org.opensearch.transport.TransportService;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import lombok.extern.log4j.Log4j2;

@Log4j2
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class TransportExecuteStreamingTaskAction extends HandledTransportAction<ActionRequest, MLExecuteTaskResponse> {
    MLTaskRunner<MLExecuteTaskRequest, MLExecuteTaskResponse> mlExecuteTaskRunner;
    TransportService transportService;

    public static StreamTransportService streamTransportService;

    public class StreamPredictActionListener<Response extends TransportResponse, Request extends TransportRequest>
        implements
            ActionListener<Response> {

        private final Request request;
        private final String actionName;
        private final ActionListener<Response> delegateListener;

        // Change the constructor signature
        public StreamPredictActionListener(ActionListener<Response> delegateListener, String actionName, Request request) {
            this.delegateListener = delegateListener;
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
    public TransportExecuteStreamingTaskAction(
        TransportService transportService,
        ActionFilters actionFilters,
        MLExecuteTaskRunner mlExecuteTaskRunner,
        StreamTransportService streamTransportService
    ) {
        super(MLExecuteStreamingTaskAction.NAME, transportService, actionFilters, MLExecuteTaskRequest::new);
        this.mlExecuteTaskRunner = mlExecuteTaskRunner;
        this.transportService = transportService;
        this.streamTransportService = streamTransportService;

        streamTransportService
            .registerRequestHandler(
                MLExecuteStreamingTaskAction.NAME,
                STREAM_PREDICT_THREAD_POOL,
                MLExecuteTaskRequest::new,
                this::messageReceived
            );
    }

    public void messageReceived(MLExecuteTaskRequest request, TransportChannel channel, Task task) {
        org.opensearch.ml.engine.algorithms.remote.StreamPredictActionListener<MLExecuteTaskResponse, MLExecuteTaskRequest> streamListener =
            new org.opensearch.ml.engine.algorithms.remote.StreamPredictActionListener<>(
                channel,
                MLExecuteStreamingTaskAction.NAME,
                request
            );
        doExecute(task, request, streamListener, channel);
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLExecuteTaskResponse> listener) {
        throw new RuntimeException();
    }

    protected void doExecute(Task task, ActionRequest request, ActionListener<MLExecuteTaskResponse> listener, TransportChannel channel) {
        MLExecuteTaskRequest mlExecuteTaskRequest = MLExecuteTaskRequest.fromActionRequest(request);
        FunctionName functionName = mlExecuteTaskRequest.getFunctionName();
        log.info("goes to transport execute stream");
        mlExecuteTaskRunner.runStream(functionName, mlExecuteTaskRequest, channel, streamTransportService, listener);
    }
}
