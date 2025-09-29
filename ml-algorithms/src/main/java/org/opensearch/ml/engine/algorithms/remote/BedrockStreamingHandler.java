/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.remote;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.transport.MLTaskResponse;

import lombok.extern.log4j.Log4j2;
import software.amazon.awssdk.http.async.SdkAsyncHttpClient;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeAsyncClient;
import software.amazon.awssdk.services.bedrockruntime.model.ContentBlock;
import software.amazon.awssdk.services.bedrockruntime.model.ContentBlockDeltaEvent;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseStreamRequest;
import software.amazon.awssdk.services.bedrockruntime.model.ConverseStreamResponseHandler;
import software.amazon.awssdk.services.bedrockruntime.model.Message;

@Log4j2
public class BedrockStreamingHandler {

    private final SdkAsyncHttpClient httpClient;

    public BedrockStreamingHandler(SdkAsyncHttpClient httpClient) {
        this.httpClient = httpClient;
    }

    public void handle(
        String action,
        Map<String, String> parameters,
        String payload,
        StreamPredictActionListener<MLTaskResponse, ?> actionListener
    ) {
        try {
            AtomicBoolean isStreamClosed = new AtomicBoolean(false);

            // Build Bedrock client
            BedrockRuntimeAsyncClient bedrockClient = buildBedrockRuntimeAsyncClient(httpClient);

            // Build request
            ConverseStreamRequest request = ConverseStreamRequest
                .builder()
                .modelId(parameters.get("model"))
                .messages(Message.builder().role("user").content(ContentBlock.builder().text(parameters.get("inputs")).build()).build())
                .build();

            // Build handler
            ConverseStreamResponseHandler handler = ConverseStreamResponseHandler.builder().onResponse(response -> {
                log.debug("Initial converse stream response: {}", response);
            }).onError(error -> {
                log.error("Converse stream error: {}", error.getMessage());
                actionListener.onFailure(new MLException("Error from remote service: " + error.getMessage(), error));
            }).onComplete(() -> {
                log.debug("Converse stream complete");
                sendCompletionResponse(isStreamClosed, actionListener);
            }).subscriber(event -> {
                log.debug("Converse stream event: {}", event);
                switch (event.sdkEventType()) {
                    case CONTENT_BLOCK_DELTA:
                        ContentBlockDeltaEvent contentEvent = (ContentBlockDeltaEvent) event;
                        String chunk = contentEvent.delta().text();
                        sendContentResponse(chunk, false, actionListener);
                        break;
                    default:
                        // Ignore other event types
                        break;
                }
            }).build();

            bedrockClient.converseStream(request, handler);

        } catch (Exception e) {
            log.error("Failed to execute Bedrock streaming", e);
            actionListener.onFailure(new MLException("Fail to execute Bedrock streaming", e));
        }
    }

    private BedrockRuntimeAsyncClient buildBedrockRuntimeAsyncClient(SdkAsyncHttpClient httpClient) {
        // Implementation to build Bedrock client
        // This would be moved from the original AwsConnectorExecutor
        return BedrockRuntimeAsyncClient.builder().httpClient(httpClient).build();
    }

    private void sendContentResponse(String content, boolean isLast, StreamPredictActionListener<MLTaskResponse, ?> actionListener) {
        // Implementation moved from AbstractConnectorExecutor
    }

    private void sendCompletionResponse(AtomicBoolean isStreamClosed, StreamPredictActionListener<MLTaskResponse, ?> actionListener) {
        // Implementation moved from AbstractConnectorExecutor
    }
}
