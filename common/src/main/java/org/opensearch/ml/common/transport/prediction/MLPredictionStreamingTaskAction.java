/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.prediction;

import org.opensearch.action.ActionType;
import org.opensearch.ml.common.transport.MLTaskResponse;

public class MLPredictionStreamingTaskAction extends ActionType<MLTaskResponse> {
    public static final MLPredictionStreamingTaskAction INSTANCE = new MLPredictionStreamingTaskAction();
    public static final String NAME = "cluster:admin/opensearch/ml/predict/stream";

    private MLPredictionStreamingTaskAction() {
        super(NAME, MLTaskResponse::new);
    }
}
