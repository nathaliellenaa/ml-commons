package org.opensearch.ml.engine.algorithms.remote;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.opensearch.rest.StreamingRestChannel;

public class StreamingRegistry {
    private static final Map<String, StreamingRestChannel> channels = new ConcurrentHashMap<>();

    public static void register(String requestId, StreamingRestChannel channel) {
        channels.put(requestId, channel);
    }

    public static StreamingRestChannel get(String requestId) {
        return channels.get(requestId);
    }

    public static void remove(String requestId) {
        channels.remove(requestId);
    }
}
