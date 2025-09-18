/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.execute;

import org.opensearch.action.ActionType;

public class MLExecuteStreamingTaskAction extends ActionType<MLExecuteTaskResponse> {
    public static final MLExecuteStreamingTaskAction INSTANCE = new MLExecuteStreamingTaskAction();
    public static final String NAME = "cluster:admin/opensearch/ml/execute/stream";

    private MLExecuteStreamingTaskAction() {
        super(NAME, MLExecuteTaskResponse::new);
    }
}
