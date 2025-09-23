/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent;

import static org.opensearch.ml.common.conversation.ActionConstants.ADDITIONAL_INFO_FIELD;
import static org.opensearch.ml.common.conversation.ActionConstants.AI_RESPONSE_FIELD;
import static org.opensearch.ml.common.utils.StringUtils.gson;
import static org.opensearch.ml.common.utils.StringUtils.processTextDoc;
import static org.opensearch.ml.common.utils.ToolUtils.filterToolOutput;
import static org.opensearch.ml.common.utils.ToolUtils.parseResponse;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.DISABLE_TRACE;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.INTERACTIONS_PREFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_CHAT_HISTORY_PREFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_PREFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_SUFFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.RESPONSE_FORMAT_INSTRUCTION;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_CALL_ID;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_RESPONSE;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.TOOL_RESULT;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.VERBOSE;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.cleanUpResource;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.constructToolParams;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.createTools;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getMcpToolSpecs;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getMessageHistoryLimit;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getMlToolSpecs;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getToolName;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getToolNames;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.outputToOutputString;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.parseLLMOutput;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.substitute;
import static org.opensearch.ml.engine.algorithms.agent.MLAgentExecutor.QUESTION;
import static org.opensearch.ml.engine.algorithms.agent.PromptTemplate.CHAT_HISTORY_PREFIX;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.security.PrivilegedActionException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import org.apache.commons.text.StringSubstitutor;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.StepListener;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.Strings;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.agent.LLMSpec;
import org.opensearch.ml.common.agent.MLAgent;
import org.opensearch.ml.common.agent.MLToolSpec;
import org.opensearch.ml.common.conversation.Interaction;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.remote.RemoteInferenceMLInput;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.memory.Memory;
import org.opensearch.ml.common.spi.memory.Message;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionStreamingTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.ml.engine.encryptor.Encryptor;
import org.opensearch.ml.engine.function_calling.FunctionCalling;
import org.opensearch.ml.engine.function_calling.FunctionCallingFactory;
import org.opensearch.ml.engine.function_calling.LLMMessage;
import org.opensearch.ml.engine.memory.ConversationIndexMemory;
import org.opensearch.ml.engine.memory.ConversationIndexMessage;
import org.opensearch.ml.engine.tools.MLModelTool;
import org.opensearch.ml.repackage.com.google.common.collect.ImmutableMap;
import org.opensearch.ml.repackage.com.google.common.collect.Lists;
import org.opensearch.remote.metadata.client.SdkClient;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.StreamTransportResponseHandler;
import org.opensearch.transport.TransportChannel;
import org.opensearch.transport.TransportException;
import org.opensearch.transport.TransportRequestOptions;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.stream.StreamTransportResponse;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;

@Log4j2
@Data
@NoArgsConstructor
public class MLChatAgentRunner implements MLAgentRunner {

    public static final String SESSION_ID = "session_id";
    public static final String LLM_TOOL_PROMPT_PREFIX = "LanguageModelTool.prompt_prefix";
    public static final String LLM_TOOL_PROMPT_SUFFIX = "LanguageModelTool.prompt_suffix";
    public static final String TOOLS = "tools";
    public static final String TOOL_DESCRIPTIONS = "tool_descriptions";
    public static final String TOOL_NAMES = "tool_names";
    public static final String OS_INDICES = "opensearch_indices";
    public static final String EXAMPLES = "examples";
    public static final String SCRATCHPAD = "scratchpad";
    public static final String CHAT_HISTORY = "chat_history";
    public static final String NEW_CHAT_HISTORY = "_chat_history";
    public static final String CONTEXT = "context";
    public static final String PROMPT = "prompt";
    public static final String LLM_RESPONSE = "llm_response";
    public static final String MAX_ITERATION = "max_iteration";
    public static final String THOUGHT = "thought";
    public static final String ACTION = "action";
    public static final String ACTION_INPUT = "action_input";
    public static final String FINAL_ANSWER = "final_answer";
    public static final String THOUGHT_RESPONSE = "thought_response";
    public static final String INTERACTIONS = "_interactions";
    public static final String INTERACTION_TEMPLATE_TOOL_RESPONSE = "interaction_template.tool_response";
    public static final String CHAT_HISTORY_QUESTION_TEMPLATE = "chat_history_template.user_question";
    public static final String CHAT_HISTORY_RESPONSE_TEMPLATE = "chat_history_template.ai_response";
    public static final String CHAT_HISTORY_MESSAGE_PREFIX = "${_chat_history.message.";
    public static final String LLM_INTERFACE = "_llm_interface";

    private static final String DEFAULT_MAX_ITERATIONS = "10";

    private Client client;
    private Settings settings;
    private ClusterService clusterService;
    private NamedXContentRegistry xContentRegistry;
    private Map<String, Tool.Factory> toolFactories;
    private Map<String, Memory.Factory> memoryFactoryMap;
    private SdkClient sdkClient;
    private Encryptor encryptor;

    public MLChatAgentRunner(
        Client client,
        Settings settings,
        ClusterService clusterService,
        NamedXContentRegistry xContentRegistry,
        Map<String, Tool.Factory> toolFactories,
        Map<String, Memory.Factory> memoryFactoryMap,
        SdkClient sdkClient,
        Encryptor encryptor
    ) {
        this.client = client;
        this.settings = settings;
        this.clusterService = clusterService;
        this.xContentRegistry = xContentRegistry;
        this.toolFactories = toolFactories;
        this.memoryFactoryMap = memoryFactoryMap;
        this.sdkClient = sdkClient;
        this.encryptor = encryptor;
    }

    @Override
    public void run(MLAgent mlAgent, Map<String, String> inputParams, ActionListener<Object> listener) {
        Map<String, String> params = new HashMap<>();
        if (mlAgent.getParameters() != null) {
            params.putAll(mlAgent.getParameters());
            for (String key : mlAgent.getParameters().keySet()) {
                if (key.startsWith("_")) {
                    params.put(key, mlAgent.getParameters().get(key));
                }
            }
        }

        params.putAll(inputParams);

        String llmInterface = params.get(LLM_INTERFACE);
        FunctionCalling functionCalling = FunctionCallingFactory.create(llmInterface);
        if (functionCalling != null) {
            functionCalling.configure(params);
        }

        String memoryType = mlAgent.getMemory().getType();
        String memoryId = params.get(MLAgentExecutor.MEMORY_ID);
        String appType = mlAgent.getAppType();
        String title = params.get(MLAgentExecutor.QUESTION);
        String chatHistoryPrefix = params.getOrDefault(PROMPT_CHAT_HISTORY_PREFIX, CHAT_HISTORY_PREFIX);
        String chatHistoryQuestionTemplate = params.get(CHAT_HISTORY_QUESTION_TEMPLATE);
        String chatHistoryResponseTemplate = params.get(CHAT_HISTORY_RESPONSE_TEMPLATE);
        int messageHistoryLimit = getMessageHistoryLimit(params);

        ConversationIndexMemory.Factory conversationIndexMemoryFactory = (ConversationIndexMemory.Factory) memoryFactoryMap.get(memoryType);
        conversationIndexMemoryFactory.create(title, memoryId, appType, ActionListener.<ConversationIndexMemory>wrap(memory -> {
            // TODO: call runAgent directly if messageHistoryLimit == 0
            memory.getMessages(ActionListener.<List<Interaction>>wrap(r -> {
                List<Message> messageList = new ArrayList<>();
                for (Interaction next : r) {
                    String question = next.getInput();
                    String response = next.getResponse();
                    // As we store the conversation with empty response first and then update when have final answer,
                    // filter out those in-flight requests when run in parallel
                    if (Strings.isNullOrEmpty(response)) {
                        continue;
                    }
                    messageList
                        .add(
                            ConversationIndexMessage
                                .conversationIndexMessageBuilder()
                                .sessionId(memory.getConversationId())
                                .question(question)
                                .response(response)
                                .build()
                        );
                }
                if (!messageList.isEmpty()) {
                    if (chatHistoryQuestionTemplate == null) {
                        StringBuilder chatHistoryBuilder = new StringBuilder();
                        chatHistoryBuilder.append(chatHistoryPrefix);
                        for (Message message : messageList) {
                            chatHistoryBuilder.append(message.toString()).append("\n");
                        }
                        params.put(CHAT_HISTORY, chatHistoryBuilder.toString());

                        // required for MLChatAgentRunnerTest.java, it requires chatHistory to be added to input params to validate
                        inputParams.put(CHAT_HISTORY, chatHistoryBuilder.toString());
                    } else {
                        List<String> chatHistory = new ArrayList<>();
                        for (Message message : messageList) {
                            Map<String, String> messageParams = new HashMap<>();
                            messageParams.put("question", processTextDoc(((ConversationIndexMessage) message).getQuestion()));

                            StringSubstitutor substitutor = new StringSubstitutor(messageParams, CHAT_HISTORY_MESSAGE_PREFIX, "}");
                            String chatQuestionMessage = substitutor.replace(chatHistoryQuestionTemplate);
                            chatHistory.add(chatQuestionMessage);

                            messageParams.clear();
                            messageParams.put("response", processTextDoc(((ConversationIndexMessage) message).getResponse()));
                            substitutor = new StringSubstitutor(messageParams, CHAT_HISTORY_MESSAGE_PREFIX, "}");
                            String chatResponseMessage = substitutor.replace(chatHistoryResponseTemplate);
                            chatHistory.add(chatResponseMessage);
                        }
                        params.put(CHAT_HISTORY, String.join(", ", chatHistory) + ", ");
                        params.put(NEW_CHAT_HISTORY, String.join(", ", chatHistory) + ", ");

                        // required for MLChatAgentRunnerTest.java, it requires chatHistory to be added to input params to validate
                        inputParams.put(CHAT_HISTORY, String.join(", ", chatHistory) + ", ");
                    }
                }

                runAgent(mlAgent, params, listener, memory, memory.getConversationId(), functionCalling);
            }, e -> {
                log.error("Failed to get chat history", e);
                listener.onFailure(e);
            }), messageHistoryLimit);
        }, listener::onFailure));
    }

    @Override
    public void runStream(MLAgent mlAgent, Map<String, String> inputParams, ActionListener<Object> listener, TransportChannel channel) {
        Map<String, String> params = new HashMap<>();
        if (mlAgent.getParameters() != null) {
            params.putAll(mlAgent.getParameters());
        }
        params.putAll(inputParams);
        params.put("stream", "true");

        String llmInterface = params.get(LLM_INTERFACE);
        FunctionCalling functionCalling = FunctionCallingFactory.create(llmInterface);
        if (functionCalling != null) {
            functionCalling.configure(params);
        }

        String memoryType = mlAgent.getMemory().getType();
        String memoryId = params.get(MLAgentExecutor.MEMORY_ID);
        String appType = mlAgent.getAppType();
        String title = params.get(QUESTION);
        String chatHistoryPrefix = params.getOrDefault(PROMPT_CHAT_HISTORY_PREFIX, CHAT_HISTORY_PREFIX);
        String chatHistoryQuestionTemplate = params.get(CHAT_HISTORY_QUESTION_TEMPLATE);
        String chatHistoryResponseTemplate = params.get(CHAT_HISTORY_RESPONSE_TEMPLATE);
        int messageHistoryLimit = getMessageHistoryLimit(params);

        ConversationIndexMemory.Factory conversationIndexMemoryFactory = (ConversationIndexMemory.Factory) memoryFactoryMap.get(memoryType);
        conversationIndexMemoryFactory.create(title, memoryId, appType, ActionListener.<ConversationIndexMemory>wrap(memory -> {
            // TODO: call runAgent directly if messageHistoryLimit == 0
            memory.getMessages(ActionListener.<List<Interaction>>wrap(r -> {
                List<Message> messageList = new ArrayList<>();
                for (Interaction next : r) {
                    String question = next.getInput();
                    String response = next.getResponse();
                    // As we store the conversation with empty response first and then update when have final answer,
                    // filter out those in-flight requests when run in parallel
                    if (Strings.isNullOrEmpty(response)) {
                        continue;
                    }
                    messageList
                        .add(
                            ConversationIndexMessage
                                .conversationIndexMessageBuilder()
                                .sessionId(memory.getConversationId())
                                .question(question)
                                .response(response)
                                .build()
                        );
                }
                if (!messageList.isEmpty()) {
                    if (chatHistoryQuestionTemplate == null) {
                        StringBuilder chatHistoryBuilder = new StringBuilder();
                        chatHistoryBuilder.append(chatHistoryPrefix);
                        for (Message message : messageList) {
                            chatHistoryBuilder.append(message.toString()).append("\n");
                        }
                        params.put(CHAT_HISTORY, chatHistoryBuilder.toString());

                        // required for MLChatAgentRunnerTest.java, it requires chatHistory to be added to input params to validate
                        inputParams.put(CHAT_HISTORY, chatHistoryBuilder.toString());
                    } else {
                        List<String> chatHistory = new ArrayList<>();
                        for (Message message : messageList) {
                            Map<String, String> messageParams = new HashMap<>();
                            messageParams.put("question", processTextDoc(((ConversationIndexMessage) message).getQuestion()));

                            StringSubstitutor substitutor = new StringSubstitutor(messageParams, CHAT_HISTORY_MESSAGE_PREFIX, "}");
                            String chatQuestionMessage = substitutor.replace(chatHistoryQuestionTemplate);
                            chatHistory.add(chatQuestionMessage);

                            messageParams.clear();
                            messageParams.put("response", processTextDoc(((ConversationIndexMessage) message).getResponse()));
                            substitutor = new StringSubstitutor(messageParams, CHAT_HISTORY_MESSAGE_PREFIX, "}");
                            String chatResponseMessage = substitutor.replace(chatHistoryResponseTemplate);
                            chatHistory.add(chatResponseMessage);
                        }
                        params.put(CHAT_HISTORY, String.join(", ", chatHistory) + ", ");
                        params.put(NEW_CHAT_HISTORY, String.join(", ", chatHistory) + ", ");

                        // required for MLChatAgentRunnerTest.java, it requires chatHistory to be added to input params to validate
                        inputParams.put(CHAT_HISTORY, String.join(", ", chatHistory) + ", ");
                    }
                }

                runAgentStream(mlAgent, params, listener, memory, memory.getConversationId(), functionCalling, channel);
            }, e -> {
                log.error("Failed to get chat history", e);
                listener.onFailure(e);
            }), messageHistoryLimit);
        }, listener::onFailure));
    }

    private final Object responseLock = new Object();
    private volatile boolean agentCompleted = false;


    private void executeStreamingRequest(
        MLPredictionTaskRequest request,
        TransportChannel channel,
        ActionListener<Object> listener,
        LLMSpec llm,
        MLAgent mlAgent,
        Map<String, Tool> tools,
        Map<String, MLToolSpec> toolSpecMap,
        Map<String, String> tmpParameters,
        List<String> interactions,
        int maxIterations,
        String tenantId,
        FunctionCalling functionCalling,
        int currentIteration,
        String sessionId,
        String parentInteractionId
    ) {
        log.info("=== executeStreamingRequest STARTED ===");
        StringBuilder accumulatedResponse = new StringBuilder();

        try {
            StreamTransportResponseHandler<MLTaskResponse> handler = new StreamTransportResponseHandler<MLTaskResponse>() {

                @Override
                public void handleStreamResponse(StreamTransportResponse<MLTaskResponse> streamResponse) {
                    log.info("=== handleStreamResponse called ===");
                    try {
                        // Process one response at a time
                        MLTaskResponse response = streamResponse.nextResponse();
                        if (response != null) {
                            log.info("Received response in AgentRunner: {}", response);

                            // Extract and log the actual content
                            String content = extractContent(response);
                            if (content != null && !content.isEmpty()) {
                                accumulatedResponse.append(content);
                            }
                            log.info("Response content: {}", content);
                            MLTaskResponse updatedResponse = addMetadataToResponse(response, sessionId, parentInteractionId);
                            channel.sendResponseBatch(updatedResponse);

                            // Recursively handle the next response - asynchronously
                            client
                                .threadPool()
                                .executor("opensearch_ml_execute_stream")
                                .execute(() -> handleStreamResponse(streamResponse));
                        } else {
                            log.info("=== STREAMING COMPLETE - Starting ReAct parsing ===");
                            log.info("Complete LLM response: {}", accumulatedResponse);

                            // Parse accumulated response for ReAct format
                            // Extract plain text from JSON chunks
                            String plainText = extractPlainTextFromChunks(accumulatedResponse.toString());
                            log.info("Extracted plain text: {}", plainText);

                            log.info("=== LLM RESPONSE ANALYSIS ===");
                            log.info("Raw LLM response: {}", plainText);
                            log.info("Function calling instance: {}", functionCalling);
                            log.info("Available tools: {}", tools.keySet());

                            // Parse for ReAct format
                            ModelTensorOutput mockOutput = null;
                            if (plainText.contains("\"choices\"") && plainText.contains("\"tool_calls\"")) {
                                // Extract clean JSON
                                String[] parts = plainText.split("\\{\"choices\":");
                                if (parts.length > 1) {
                                    String cleanJson = "{\"choices\":" + parts[parts.length - 1];

                                    // Use function calling mock output (no "response" wrapper)
                                    mockOutput = createMockOutput(cleanJson);
                                }
                            } else {
                                // Use regular mock output (with "response" wrapper for ReAct)
                                mockOutput = createMockFinalAnswerOutput(plainText);
                                // This will go to the if block (ReAct path)
                            }

                            // ModelTensorOutput mockOutput = createMockOutput(plainText);
                            List<String> llmResponsePatterns = gson.fromJson(tmpParameters.get("llm_response_pattern"), List.class);
                            log.info("Mock output: {}", mockOutput);
                            log.info("llmResponsePatterns: {}", llmResponsePatterns);
                            Map<String, String> parsed = parseLLMOutput(
                                tmpParameters,
                                mockOutput,
                                llmResponsePatterns,
                                tools.keySet(),
                                interactions,
                                functionCalling
                            );

                            if (parsed.containsKey(ACTION) && !interactions.isEmpty()) {
                                try {
                                    String lastInteraction = interactions.get(interactions.size() - 1);
                                    Map<String, Object> messageMap = gson.fromJson(lastInteraction, Map.class);

                                    // If it's missing the role field, add it
                                    if (!messageMap.containsKey("role") && messageMap.containsKey("tool_calls")) {
                                        messageMap.put("role", "assistant");
                                        interactions.set(interactions.size() - 1, StringUtils.toJson(messageMap));
                                        log.info("Fixed assistant message role in interactions");
                                    }
                                } catch (Exception e) {
                                    log.error("Failed to fix assistant message role", e);
                                }
                            }

                            log.info("Parsed keys: {}", parsed.keySet());
                            log.info("Parsed values: {}", parsed);
                            log.info("ACTION found: {}", parsed.get(ACTION));
                            log.info("FINAL_ANSWER found: {}", parsed.get(FINAL_ANSWER));

                            if (parsed.containsKey(ACTION) || parsed.get("thought_response").contains("Action:")) {
                                log
                                    .info(
                                        "Found Action in response - executing tool: {}",
                                        parsed.get("thought_response").contains("Action:")
                                    );
                                executeReActLoopWithStreaming(
                                    llm,
                                    mlAgent,
                                    tools,
                                    toolSpecMap,
                                    tmpParameters,
                                    interactions,
                                    maxIterations,
                                    tenantId,
                                    listener,
                                    functionCalling,
                                    channel,
                                    currentIteration + 1,
                                    parsed,
                                    sessionId,
                                    parentInteractionId

                                );
                            } else {
                                log.info("No Action found - treating as final answer");
                                channel.completeStream();
                            }
                            streamResponse.close();
                            // log.info("No more responses in agent, closing stream");
                            // // channel.sendChunk(XContentHttpChunk.last());
                            // channel.completeStream();
                            // streamResponse.close();
                        }
                    } catch (Exception e) {
                        streamResponse.cancel("Error processing stream", e);
                        log.error("Error in stream handling", e);
                    }
                }

                @Override
                public void handleException(TransportException exp) {
                    listener.onFailure(exp);
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

            // Use reflection to access streamTransportService
            Class<?> transportActionClass = Class.forName("org.opensearch.ml.action.prediction.TransportPredictionStreamingTaskAction");
            Field streamServiceField = transportActionClass.getDeclaredField("streamTransportService");
            streamServiceField.setAccessible(true);

            Object streamTransportService = streamServiceField.get(null);

            if (streamTransportService == null) {
                throw new IllegalStateException("StreamTransportService not available");
            }

            Class<?> streamServiceClass = streamTransportService.getClass();
            Method sendRequestMethod = streamServiceClass
                .getMethod(
                    "sendRequest",
                    org.opensearch.cluster.node.DiscoveryNode.class,
                    String.class,
                    org.opensearch.transport.TransportRequest.class,
                    TransportRequestOptions.class,
                    org.opensearch.transport.TransportResponseHandler.class
                );

            sendRequestMethod
                .invoke(
                    streamTransportService,
                    clusterService.localNode(),
                    MLPredictionStreamingTaskAction.NAME,
                    request,
                    TransportRequestOptions.builder().withType(TransportRequestOptions.Type.STREAM).build(),
                    handler
                );

        } catch (Exception e) {
            log.error("Failed to execute streaming request", e);
            listener.onFailure(e);
        }
    }

//    private ModelTensorOutput createMockOutput(String functionCallJson) {
//        try {
//            // Extract the final complete JSON (the one with "choices")
//            String cleanJson = extractFinalJson(functionCallJson);
//            log.info("Extracted clean JSON: {}", cleanJson);
//
//            Map<String, Object> parsedJson = gson.fromJson(cleanJson, Map.class);
//
//            ModelTensor tensor = ModelTensor
//                    .builder()
//                    .name("function_call_response")
//                    .dataAsMap(parsedJson)
//                    .build();
//
//            ModelTensors tensors = ModelTensors.builder().mlModelTensors(List.of(tensor)).build();
//            return ModelTensorOutput.builder().mlModelOutputs(List.of(tensors)).build();
//
//        } catch (Exception e) {
//            log.error("Failed to parse function calling JSON: {}", functionCallJson, e);
//            return createMockFinalAnswerOutput(functionCallJson);
//        }
//    }

    private ModelTensorOutput createMockOutput(String functionCallJson) {
        try {
            String trimmed = functionCallJson.trim();
            if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
                return createMockFinalAnswerOutput(functionCallJson);
            }

            // CRITICAL FIX: Extract only the final valid JSON object
            String validJson = extractValidJson(functionCallJson);
            if (validJson == null) {
                return createMockFinalAnswerOutput(functionCallJson);
            }

            Map<String, Object> parsedJson = gson.fromJson(validJson, Map.class);

            ModelTensor tensor = ModelTensor
                    .builder()
                    .name("response")
                    .dataAsMap(parsedJson)
                    .build();

            ModelTensors tensors = ModelTensors.builder().mlModelTensors(List.of(tensor)).build();
            return ModelTensorOutput.builder().mlModelOutputs(List.of(tensors)).build();

        } catch (Exception e) {
            log.error("Failed to parse function calling JSON: {}", functionCallJson, e);
            return createMockFinalAnswerOutput(functionCallJson);
        }
    }

    private String extractValidJson(String concatenatedJson) {
        // Find the last occurrence of {"choices": which should be the valid response
        int lastIndex = concatenatedJson.lastIndexOf("{\"choices\":");
        if (lastIndex == -1) {
            return null;
        }

        // Extract from that point to the end and find the matching closing brace
        String candidate = concatenatedJson.substring(lastIndex);
        int braceCount = 0;
        int endIndex = -1;

        for (int i = 0; i < candidate.length(); i++) {
            char c = candidate.charAt(i);
            if (c == '{') braceCount++;
            else if (c == '}') {
                braceCount--;
                if (braceCount == 0) {
                    endIndex = i + 1;
                    break;
                }
            }
        }

        if (endIndex != -1) {
            String result = candidate.substring(0, endIndex);
            log.info("Extracted valid JSON: {}", result);
            return result;
        }

        return null;
    }



    // private ModelTensorOutput createMockOutput(String functionCallJson) {
    // try {
    // // Parse the function calling JSON
    // Map<String, Object> parsedJson = gson.fromJson(functionCallJson, Map.class);
    //
    // // Create tensor with parsed JSON as the direct dataAsMap (no nesting)
    // ModelTensor tensor = ModelTensor
    // .builder()
    // .name("function_call_response") // Different name
    // .dataAsMap(parsedJson) // Direct structure: {choices: [...]}
    // .build();
    //
    // ModelTensors tensors = ModelTensors.builder().mlModelTensors(List.of(tensor)).build();
    // return ModelTensorOutput.builder().mlModelOutputs(List.of(tensors)).build();
    //
    // } catch (Exception e) {
    // log.error("Failed to parse function calling JSON", e);
    // return createMockFinalAnswerOutput(functionCallJson);
    // }
    // }

    private MLTaskResponse addMetadataToResponse(MLTaskResponse response, String sessionId, String parentInteractionId) {
        if (response == null || response.getOutput() == null)
            return response;

        ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
        List<ModelTensors> updatedOutputs = new ArrayList<>();

        for (ModelTensors tensors : output.getMlModelOutputs()) {
            List<ModelTensor> updatedTensors = new ArrayList<>();

            // Add memory_id and parent_interaction_id tensors
            updatedTensors.add(ModelTensor.builder().name("memory_id").result(sessionId).build());
            updatedTensors.add(ModelTensor.builder().name("parent_interaction_id").result(parentInteractionId).build());

            // Process existing tensors and FORCE is_last to false
            for (ModelTensor tensor : tensors.getMlModelTensors()) {
                if ("llm_response".equals(tensor.getName()) && tensor.getDataAsMap() != null) {
                    Map<String, Object> dataMap = new HashMap<>(tensor.getDataAsMap());
                    // Only force is_last=false for empty content during agent execution (not at true end)
                    String content = (String) dataMap.get("content");
                    boolean isLast = Boolean.TRUE.equals(dataMap.get("is_last"));

                    if (isLast && (content == null || content.trim().isEmpty()) && !agentCompleted) {
                        // This is an intermediate LLM completion, not the final answer
                        dataMap.put("is_last", false);
                    }
                    updatedTensors.add(ModelTensor.builder().name(tensor.getName()).dataAsMap(dataMap).build());
                } else {
                    updatedTensors.add(tensor);
                }
            }

            updatedOutputs.add(ModelTensors.builder().mlModelTensors(updatedTensors).build());
        }

        ModelTensorOutput updatedOutput = ModelTensorOutput.builder().mlModelOutputs(updatedOutputs).build();
        return new MLTaskResponse(updatedOutput);
    }

    private ModelTensorOutput createMockFinalAnswerOutput(String finalAnswerText) {
        // Create the structure that LLM_RESPONSE_FILTER expects: $.choices[0].message.content
        Map<String, Object> message = Map.of("content", finalAnswerText);
        Map<String, Object> choice = Map
            .of(
                "message",
                message,
                "finish_reason",
                "stop"  // **ADD THIS - parseLLMOutput checks this**
            );
        Map<String, Object> response = Map.of("choices", List.of(choice));

        ModelTensor tensor = ModelTensor
            .builder()
            .name("final_answer_response")
            .dataAsMap(response)  // This will have choices[0].message.content
            .build();

        ModelTensors tensors = ModelTensors.builder().mlModelTensors(List.of(tensor)).build();
        return ModelTensorOutput.builder().mlModelOutputs(List.of(tensors)).build();
    }

    private String extractPlainTextFromChunks(String jsonChunks) {
        StringBuilder plainText = new StringBuilder();
        try {
            // Split by }{ to separate JSON objects
            String[] chunks = jsonChunks.split("\\}\\{");
            for (int i = 0; i < chunks.length; i++) {
                String chunk = chunks[i];
                if (i > 0)
                    chunk = "{" + chunk;
                if (i < chunks.length - 1)
                    chunk = chunk + "}";

                JsonObject json = JsonParser.parseString(chunk).getAsJsonObject();
                if (json.has("content")) {
                    String content = json.get("content").getAsString();
                    plainText.append(content);
                }
            }
        } catch (Exception e) {
            log.error("Failed to extract plain text from chunks", e);
            return jsonChunks; // Fallback to original
        }
        return plainText.toString();
    }

    private void runAgentStream(
        MLAgent mlAgent,
        Map<String, String> params,
        ActionListener<Object> listener,
        Memory memory,
        String sessionId,
        FunctionCalling functionCalling,
        TransportChannel channel
    ) {
        List<MLToolSpec> toolSpecs = getMlToolSpecs(mlAgent, params);

        // Create a common method to handle both success and failure cases
        Consumer<List<MLToolSpec>> processTools = (allToolSpecs) -> {
            Map<String, Tool> tools = new HashMap<>();
            Map<String, MLToolSpec> toolSpecMap = new HashMap<>();
            createTools(toolFactories, params, allToolSpecs, tools, toolSpecMap, mlAgent);
            runReActStream(
                mlAgent.getLlm(),
                mlAgent,
                tools,
                toolSpecMap,
                params,
                memory,
                sessionId,
                mlAgent.getTenantId(),
                listener,
                functionCalling,
                channel
            );
        };

        // Fetch MCP tools and handle both success and failure cases
        getMcpToolSpecs(mlAgent, client, sdkClient, encryptor, ActionListener.wrap(mcpTools -> {
            toolSpecs.addAll(mcpTools);
            processTools.accept(toolSpecs);
        }, e -> {
            log.error("Failed to get MCP tools, continuing with base tools only", e);
            processTools.accept(toolSpecs);
        }));
    }

    private void runReActStream(
        LLMSpec llm,
        MLAgent mlAgent,
        Map<String, Tool> tools,
        Map<String, MLToolSpec> toolSpecMap,
        Map<String, String> parameters,
        Memory memory,
        String sessionId,
        String tenantId,
        ActionListener<Object> listener,
        FunctionCalling functionCalling,
        TransportChannel channel
    ) {
        Map<String, String> tmpParameters = constructLLMParams(llm, parameters);
        String prompt = constructLLMPrompt(tools, tmpParameters);
        tmpParameters.put(PROMPT, prompt);
        final String finalPrompt = prompt;

        String question = tmpParameters.get(QUESTION);
        String parentInteractionId = tmpParameters.get(MLAgentExecutor.PARENT_INTERACTION_ID);
        boolean verbose = Boolean.parseBoolean(tmpParameters.getOrDefault(VERBOSE, "false"));
        boolean traceDisabled = tmpParameters.containsKey(DISABLE_TRACE) && Boolean.parseBoolean(tmpParameters.get(DISABLE_TRACE));

        // Create root interaction.
        ConversationIndexMemory conversationIndexMemory = (ConversationIndexMemory) memory;

        // Trace number
        AtomicInteger traceNumber = new AtomicInteger(0);

        AtomicReference<StepListener<MLTaskResponse>> lastLlmListener = new AtomicReference<>();
        AtomicReference<String> lastThought = new AtomicReference<>();
        AtomicReference<String> lastAction = new AtomicReference<>();
        AtomicReference<String> lastActionInput = new AtomicReference<>();
        AtomicReference<String> lastToolSelectionResponse = new AtomicReference<>();
        Map<String, Object> additionalInfo = new ConcurrentHashMap<>();

        StepListener firstListener = new StepListener<MLTaskResponse>();
        log.info("=== CREATED FIRST LISTENER: {} ===", firstListener);
        lastLlmListener.set(firstListener);
        StepListener<?> lastStepListener = firstListener;

        StringBuilder scratchpadBuilder = new StringBuilder();
        List<String> interactions = new CopyOnWriteArrayList<>();

        StringSubstitutor tmpSubstitutor = new StringSubstitutor(Map.of(SCRATCHPAD, scratchpadBuilder.toString()), "${parameters.", "}");
        AtomicReference<String> newPrompt = new AtomicReference<>(tmpSubstitutor.replace(prompt));
        tmpParameters.put(PROMPT, newPrompt.get());

        List<ModelTensors> traceTensors = createModelTensors(sessionId, parentInteractionId);
        int maxIterations = Integer.parseInt(tmpParameters.getOrDefault(MAX_ITERATION, DEFAULT_MAX_ITERATIONS));
        Map<String, String> currentModelOutput = null;

        for (int i = 0; i < maxIterations; i++) {
            int finalI = i;
            StepListener<?> nextStepListener = new StepListener<>();

            lastStepListener.whenComplete(output -> {
                StringBuilder sessionMsgAnswerBuilder = new StringBuilder();
                log.info("ReAct loop iteration {}, output type: {}", finalI, output.getClass().getSimpleName());

                if (finalI % 2 == 0) {
                    log.info("Processing LLM response in ReAct loop");

                    MLTaskResponse llmResponse = (MLTaskResponse) output;
                    ModelTensorOutput tmpModelTensorOutput = (ModelTensorOutput) llmResponse.getOutput();
                    List<String> llmResponsePatterns = gson.fromJson(tmpParameters.get("llm_response_pattern"), List.class);
                    Map<String, String> modelOutput = parseLLMOutput(
                        parameters,
                        tmpModelTensorOutput,
                        llmResponsePatterns,
                        tools.keySet(),
                        interactions,
                        functionCalling
                    );

                    if (!interactions.isEmpty()) {
                        try {
                            String lastInteraction = interactions.get(interactions.size() - 1);
                            Map<String, Object> messageMap = gson.fromJson(lastInteraction, Map.class);

                            if (!messageMap.containsKey("role") && messageMap.containsKey("tool_calls")) {
                                messageMap.put("role", "assistant");
                                interactions.set(interactions.size() - 1, StringUtils.toJson(messageMap));
                                log.info("Fixed assistant message role in interactions after parseLLMOutput");
                            }
                        } catch (Exception e) {
                            log.error("Failed to fix assistant message role after parseLLMOutput", e);
                        }
                    }

                    String thought = String.valueOf(modelOutput.get(THOUGHT));
                    String toolCallId = String.valueOf(modelOutput.get("tool_call_id"));
                    String action = String.valueOf(modelOutput.get(ACTION));
                    String actionInput = String.valueOf(modelOutput.get(ACTION_INPUT));
                    String thoughtResponse = modelOutput.get(THOUGHT_RESPONSE);
                    String finalAnswer = modelOutput.get(FINAL_ANSWER);

                    if (finalAnswer != null) {
                        finalAnswer = finalAnswer.trim();
                        MLTaskResponse finalChunk = createFinalChunk(finalAnswer, sessionId, parentInteractionId);
//                        try {
//                            channel.sendResponseBatch(finalChunk);
//                        } catch (Exception e) {
//                            log.warn("Channel already closed, skipping final chunk: {}", e.getMessage());
//                        }
                        // Don't send accumulated response in streaming mode - just save to memory
                        saveToMemoryOnly(
                            sessionId,
                            listener,
                            question,
                            parentInteractionId,
                            verbose,
                            traceDisabled,
                            traceTensors,
                            conversationIndexMemory,
                            traceNumber,
                            additionalInfo,
                            finalAnswer,
                            channel
                        );
                        cleanUpResource(tools);
                        return;
                    }
                    sessionMsgAnswerBuilder.append(thought);
                    lastThought.set(thought);
                    lastAction.set(action);
                    lastActionInput.set(actionInput);
                    lastToolSelectionResponse.set(thoughtResponse);

                    traceTensors
                        .add(
                            ModelTensors
                                .builder()
                                .mlModelTensors(List.of(ModelTensor.builder().name("response").result(thoughtResponse).build()))
                                .build()
                        );

                    saveTraceData(
                        conversationIndexMemory,
                        memory.getType(),
                        question,
                        thoughtResponse,
                        sessionId,
                        traceDisabled,
                        parentInteractionId,
                        traceNumber,
                        "LLM"
                    );

                    if (tools.containsKey(action)) {
                        Map<String, String> toolParams = constructToolParams(
                            tools,
                            toolSpecMap,
                            question,
                            lastActionInput,
                            action,
                            actionInput
                        );
                        runTool(
                            tools,
                            toolSpecMap,
                            tmpParameters,
                            (ActionListener<Object>) nextStepListener,
                            action,
                            actionInput,
                            toolParams,
                            interactions,
                            toolCallId,
                            functionCalling
                        );
                    } else {
                        String res = String.format(Locale.ROOT, "Failed to run the tool %s which is unsupported.", action);
                        StringSubstitutor substitutor = new StringSubstitutor(
                            Map.of(SCRATCHPAD, scratchpadBuilder.toString()),
                            "${parameters.",
                            "}"
                        );
                        newPrompt.set(substitutor.replace(finalPrompt));
                        tmpParameters.put(PROMPT, newPrompt.get());
                        ((ActionListener<Object>) nextStepListener).onResponse(res);
                    }
                } else {
                    log.info("Processing tool execution in ReAct loop");

                    addToolOutputToAddtionalInfo(toolSpecMap, lastAction, additionalInfo, output);
                    String toolResponse = constructToolResponse(
                        tmpParameters,
                        lastAction,
                        lastActionInput,
                        lastToolSelectionResponse,
                        output
                    );
                    scratchpadBuilder.append(toolResponse).append("\n\n");

                    saveTraceData(
                        conversationIndexMemory,
                        "ReAct",
                        lastActionInput.get(),
                        outputToOutputString(output),
                        sessionId,
                        traceDisabled,
                        parentInteractionId,
                        traceNumber,
                        lastAction.get()
                    );

                    StringSubstitutor substitutor = new StringSubstitutor(Map.of(SCRATCHPAD, scratchpadBuilder), "${parameters.", "}");
                    newPrompt.set(substitutor.replace(finalPrompt));
                    tmpParameters.put(PROMPT, newPrompt.get());
                    if (interactions.size() > 0) {
                        tmpParameters.put(INTERACTIONS, ", " + String.join(", ", interactions));
                    }

                    sessionMsgAnswerBuilder.append(outputToOutputString(output));

                    MLTaskResponse toolChunk = createToolResponseChunk(outputToOutputString(output), sessionId, parentInteractionId);
                    log.info("=== TOOL RESPONSE CHUNK DEBUG ===");
                    log.info("Tool output string: {}", outputToOutputString(output));
                    log.info("Tool chunk created: {}", toolChunk);
                    log.info("Tool chunk content: {}", extractContent(toolChunk));


                    try {
                        channel.sendResponseBatch(toolChunk);
                        log.info("Successfully sent tool response chunk");
                    } catch (Exception e) {
                        log.error("Failed to send tool response chunk", e);
                    }

                    traceTensors
                        .add(
                            ModelTensors
                                .builder()
                                .mlModelTensors(
                                    Collections
                                        .singletonList(
                                            ModelTensor.builder().name("response").result(sessionMsgAnswerBuilder.toString()).build()
                                        )
                                )
                                .build()
                        );

                    if (finalI == maxIterations - 1) {
                        // Send final chunk with is_last=true
                        String finalResponse = verbose ? "Max iterations reached" : lastThought.get();
                        MLTaskResponse finalChunk = createFinalChunk(finalResponse, sessionId, parentInteractionId);
                        try {
                            channel.sendResponseBatch(finalChunk);
                        } catch (Exception e) {
                            log.warn("Channel already closed, skipping final chunk: {}", e.getMessage());
                        }

                        if (verbose) {
                            listener.onResponse(ModelTensorOutput.builder().mlModelOutputs(traceTensors).build());
                        } else {
                            List<ModelTensors> finalModelTensors = createFinalAnswerTensors(
                                createModelTensors(sessionId, parentInteractionId),
                                List.of(ModelTensor.builder().name("response").dataAsMap(Map.of("response", lastThought.get())).build())
                            );
                            listener.onResponse(ModelTensorOutput.builder().mlModelOutputs(finalModelTensors).build());
                        }
                    } else {
//                        String lastInteraction = interactions.get(interactions.size() - 1);
//                        Map<String, Object> messageMap = gson.fromJson(lastInteraction, Map.class);
//
//                        // If it's missing the role field, add it
//                        if (!messageMap.containsKey("role") && messageMap.containsKey("tool_calls")) {
//                            messageMap.put("role", "assistant");
//                            interactions.set(interactions.size() - 1, StringUtils.toJson(messageMap));
//                            log.info("Fixed assistant message role in interactions");
//                        }
                        ActionRequest request = new MLPredictionTaskRequest(
                            llm.getModelId(),
                            RemoteInferenceMLInput
                                .builder()
                                .algorithm(FunctionName.REMOTE)
                                .inputDataset(RemoteInferenceInputDataSet.builder().parameters(tmpParameters).build())
                                .build(),
                            null,
                            tenantId
                        );
                        executeStreamingLLMCall(
                            request,
                            (ActionListener<MLTaskResponse>) nextStepListener,
                            sessionId,
                            parentInteractionId,
                            channel
                        );
                    }
                }
            }, e -> {
                log.error("Failed to run chat agent", e);
                listener.onFailure(e);
            });
            if (i < maxIterations - 1) {
                lastStepListener = nextStepListener;
            }

            // Execute ReAct loop with streaming support
            // executeReActLoopWithStreaming(
            // llm,
            // mlAgent,
            // tools,
            // toolSpecMap,
            // tmpParameters,
            // interactions,
            // maxIterations,
            // tenantId,
            // listener,
            // functionCalling,
            // channel,
            // 0,
            // currentModelOutput,
            // sessionId,
            // parentInteractionId
            // );
        }
        // After the loop, trigger the first iteration by calling the first listener directly
        ActionRequest firstRequest = new MLPredictionTaskRequest(
                llm.getModelId(),
                RemoteInferenceMLInput.builder()
                        .algorithm(FunctionName.REMOTE)
                        .inputDataset(RemoteInferenceInputDataSet.builder().parameters(tmpParameters).build())
                        .build(),
                null,
                tenantId
        );
        executeStreamingLLMCall(firstRequest, firstListener, sessionId, parentInteractionId, channel);

    }

    // In your final answer handling or when max iterations reached
    private MLTaskResponse createFinalChunk(String finalAnswer, String sessionId, String parentInteractionId) {
        List<ModelTensor> tensors = Arrays.asList(
                ModelTensor.builder().name("memory_id").result(sessionId).build(),
                ModelTensor.builder().name("parent_interaction_id").result(parentInteractionId).build(),
                ModelTensor.builder()
                        .name("response")
                        .dataAsMap(Map.of(
                                "content", finalAnswer,
                                "is_last", true  // Only final answers are last
                        ))
                        .build()
        );

        ModelTensors modelTensors = ModelTensors.builder().mlModelTensors(tensors).build();
        ModelTensorOutput output = ModelTensorOutput.builder().mlModelOutputs(List.of(modelTensors)).build();
        return new MLTaskResponse(output);
    }

    private void executeStreamingLLMCall(
        ActionRequest request,
        ActionListener<MLTaskResponse> nextStepListener,
        String sessionId,
        String parentInteractionId,
        TransportChannel channel
    ) {
        try {
            StreamTransportResponseHandler<MLTaskResponse> handler = createStreamingHandler(
                nextStepListener,
                sessionId,
                parentInteractionId,
                channel
            );

            // Use reflection to get streamTransportService
            Class<?> transportActionClass = Class.forName("org.opensearch.ml.action.prediction.TransportPredictionStreamingTaskAction");
            Field streamServiceField = transportActionClass.getDeclaredField("streamTransportService");
            streamServiceField.setAccessible(true);
            Object streamTransportService = streamServiceField.get(null);

            Method sendRequestMethod = streamTransportService
                .getClass()
                .getMethod(
                    "sendRequest",
                    org.opensearch.cluster.node.DiscoveryNode.class,
                    String.class,
                    org.opensearch.transport.TransportRequest.class,
                    TransportRequestOptions.class,
                    org.opensearch.transport.TransportResponseHandler.class
                );

            sendRequestMethod
                .invoke(
                    streamTransportService,
                    clusterService.localNode(),
                    MLPredictionStreamingTaskAction.NAME,
                    request,
                    TransportRequestOptions.builder().withType(TransportRequestOptions.Type.STREAM).build(),
                    handler
                );
        } catch (Exception e) {
            nextStepListener.onFailure(e);
        }
    }

    private MLTaskResponse createToolResponseChunk(String toolOutput, String sessionId, String parentInteractionId) {
        List<ModelTensor> tensors = Arrays.asList(
                ModelTensor.builder()
                        .name("response")
                        .dataAsMap(Map.of(
                                "content", toolOutput,
                                "is_last", false  // Tool responses are never last
                        ))
                        .build(),
                ModelTensor.builder().name("memory_id").result(sessionId).build(),
                ModelTensor.builder().name("parent_interaction_id").result(parentInteractionId).build()
        );

        ModelTensors modelTensors = ModelTensors.builder().mlModelTensors(tensors).build();
        ModelTensorOutput output = ModelTensorOutput.builder().mlModelOutputs(List.of(modelTensors)).build();
        return new MLTaskResponse(output);
    }

    private StreamTransportResponseHandler<MLTaskResponse> createStreamingHandler(
        ActionListener<MLTaskResponse> nextStepListener,
        String sessionId,
        String parentInteractionId,
        TransportChannel channel
    ) {
        return new StreamTransportResponseHandler<MLTaskResponse>() {
            private StringBuilder accumulatedResponse = new StringBuilder();

            @Override
            public void handleStreamResponse(StreamTransportResponse<MLTaskResponse> streamResponse) {
                log.info("=== STREAMING HANDLER CALLED ===");
                try {
                    MLTaskResponse response;
                    while ((response = streamResponse.nextResponse()) != null) {
                        // Add metadata and forward to channel
//                        MLTaskResponse updatedResponse = addMetadataToResponse(response, sessionId, parentInteractionId);
//                        channel.sendResponseBatch(updatedResponse);

                        // Accumulate response content
                        String content = extractContent(response);

                        // Skip empty responses - they're just LLM completion markers
                        if (content == null || content.trim().isEmpty()) {
                            log.info("Skipping empty response");
                            continue;
                        }

                        // Add metadata and forward to channel
                        MLTaskResponse updatedResponse = addMetadataToResponse(response, sessionId, parentInteractionId);
                        channel.sendResponseBatch(updatedResponse);

                        // Accumulate response content
                        accumulatedResponse.append(content);
//                        if (content != null && !content.isEmpty()) {
//                            accumulatedResponse.append(content);
//                        }

                        // Check if this is the last response (empty content usually indicates end)
//                        if (content == null || content.isEmpty()) {
//                            log.info("Received empty content, assuming stream end");
//                            break;
//                        }
                    }

                    // Stream is complete - create final response and continue ReAct loop
                    String fullResponse = accumulatedResponse.toString();
                    log.info("=== CALLING nextStepListener.onResponse ===");
                    log.info("Full response: {}", fullResponse);

                    ModelTensorOutput mockOutput = createMockOutput(fullResponse);
                    MLTaskResponse finalResponse = new MLTaskResponse(mockOutput);

                    // THIS IS THE KEY FIX - call nextStepListener immediately when stream completes
                    log.info("About to call nextStepListener.onResponse with: {}", finalResponse);
                    nextStepListener.onResponse(finalResponse);
                    log.info("=== CALLED nextStepListener.onResponse ===");

                    streamResponse.close();

                } catch (Exception e) {
                    streamResponse.cancel("Error processing stream", e);
                    nextStepListener.onFailure(e);
                }
            }

            @Override
            public void handleException(TransportException exp) {
                nextStepListener.onFailure(exp);
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
    }

    private String extractContent(MLTaskResponse response) {
        log.info("=== EXTRACT CONTENT DEBUG ===");
        try {
            ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
            log.info("ModelTensorOutput: {}", output);
            if (output != null && !output.getMlModelOutputs().isEmpty()) {
                log.info("Number of mlModelOutputs: {}", output.getMlModelOutputs().size());
                ModelTensors modelTensors = output.getMlModelOutputs().get(0);
                log.info("ModelTensors: {}", modelTensors);
                log.info("Number of mlModelTensors: {}", modelTensors.getMlModelTensors().size());
                if (!modelTensors.getMlModelTensors().isEmpty()) {
                    Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(0).getDataAsMap();
                    if (dataMap != null && dataMap.containsKey("content")) {
                        String content = (String) dataMap.get("content");
                        log.info("Found content: '{}'", content);
                        return (String) dataMap.get("content");
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to extract content", e);
        }
        return "";
    }

    private void executeReActLoopWithStreaming(
        LLMSpec llm,
        MLAgent mlAgent,
        Map<String, Tool> tools,
        Map<String, MLToolSpec> toolSpecMap,
        Map<String, String> tmpParameters,
        List<String> interactions,
        int maxIterations,
        String tenantId,
        ActionListener<Object> listener,
        FunctionCalling functionCalling,
        TransportChannel channel,
        int currentIteration,
        Map<String, String> currentModelOutput,
        String sessionId,
        String parentInteractionId
    ) {
        log.info("goes to executeReActLoopWithStreaming");
        // log.info("at iteration {}, tmpParams {}", currentIteration, tmpParameters);
        if (currentIteration >= maxIterations) {
            // Max iterations reached - return current state
            listener.onResponse("Max iterations reached");
            return;
        }

        if (!interactions.isEmpty()) {
            tmpParameters.put(INTERACTIONS, ", " + String.join(", ", interactions));
            log.info("Built _interactions parameter: {}", tmpParameters.get(INTERACTIONS));
        }

        if (currentIteration % 2 == 0) {
            log.info("=== STREAMING LLM RESPONSE ===");
            log.info("=== PARAMETERS SENT TO LLM ===");
            log.info("Interactions history: {}", interactions);
            log.info("Scratchpad content: {}", tmpParameters.get(SCRATCHPAD));
            log.info("All parameters: {}", tmpParameters.keySet());
            log.info("chat_history: {}", tmpParameters.get(CHAT_HISTORY));
            log.info("_chat_history: {}", tmpParameters.get(NEW_CHAT_HISTORY));
            log.info("Chat history template user question: {}", tmpParameters.get("chat_history_template.user_question"));
            log.info("Chat history template AI response: {}", tmpParameters.get("chat_history_template.ai_response"));

            // Fix - accumulate complete response for ReAct parsing
            StringBuilder accumulatedResponse = new StringBuilder();
            String modelId = mlAgent.getLlm().getModelId();
            MLInput mlInput = buildMLInputFromParams(tmpParameters);
            MLPredictionTaskRequest request = MLPredictionTaskRequest.builder().mlInput(mlInput).modelId(modelId).build();
            executeStreamingRequest(
                request,
                channel,
                listener,
                llm,
                mlAgent,
                tools,
                toolSpecMap,
                tmpParameters,
                interactions,
                maxIterations,
                tenantId,
                functionCalling,
                currentIteration,
                sessionId,
                parentInteractionId
            );
        } else {
            // Tool execution step - NEVER STREAM (use regular execution)
            log.info("=== TOOL EXECUTION (NON-STREAMING) ===");

            if (currentModelOutput == null) {
                listener.onFailure(new IllegalStateException("No LLM output available for tool execution"));
                return;
            }

            String action = currentModelOutput.get(ACTION);
            String actionInput = currentModelOutput.get(ACTION_INPUT);

            String thoughtResponse = currentModelOutput.get("thought_response");
            // log.info("ACTION is null, extracting from thought_response: {}", thoughtResponse);

            if (thoughtResponse != null) {
                try {
                    // Check if it contains function calling JSON
                    if (thoughtResponse.contains("\"choices\"") && thoughtResponse.contains("\"tool_calls\"")) {
                        // Extract from the final JSON part (after the chunks)
                        String[] parts = thoughtResponse.split("\\{\"choices\":");
                        if (parts.length > 1) {
                            String jsonPart = "{\"choices\":" + parts[parts.length - 1];
                            JsonObject response = JsonParser.parseString(jsonPart).getAsJsonObject();

                            JsonArray choices = response.getAsJsonArray("choices");
                            if (choices.size() > 0) {
                                JsonObject choice = choices.get(0).getAsJsonObject();
                                JsonObject message = choice.getAsJsonObject("message");
                                JsonArray toolCalls = message.getAsJsonArray("tool_calls");

                                if (toolCalls.size() > 0) {
                                    JsonObject toolCall = toolCalls.get(0).getAsJsonObject();
                                    JsonObject function = toolCall.getAsJsonObject("function");

                                    action = function.get("name").getAsString(); // "RetrieveIndexMetaTool"
                                    actionInput = function.get("arguments").getAsString(); // "{}"

                                    log.info("Extracted action: {}, actionInput: {}", action, actionInput);
                                }
                            }
                        }
                    }
                    // Fallback to original ReAct parsing if no function calls found
                    else if (thoughtResponse.contains("Action:")) {
                        // Your original ReAct parsing logic here...
                    }

                } catch (Exception e) {
                    log.error("Failed to parse function calling response", e);
                    // Fallback to original parsing
                }
            }

            log.info("action here is {}", action);
            log.info("action input here is {}", actionInput);

            String finalAction = action;
            String finalActionInput = actionInput;

            // Execute tool using regular (non-streaming) approach
            Map<String, String> toolParams = constructToolParams(
                tools,
                toolSpecMap,
                tmpParameters.get(QUESTION),
                new AtomicReference<>(actionInput),
                action,
                actionInput
            );

            if (actionInput != null && !actionInput.trim().startsWith("{")) {
                // Convert plain string to JSON format expected by tool
                String jsonInput = String.format("{\"input\": \"%s\"}", actionInput.replace("\"", "\\\""));
                toolParams.put("input", jsonInput);
                log.info("Converted plain string to JSON: {}", jsonInput);
            }
            String toolCallId = String.valueOf(currentModelOutput.get("tool_call_id"));
            runTool(tools, toolSpecMap, tmpParameters, ActionListener.wrap(toolResult -> {
                // Convert tool result to single chunk and send
                // sendToolResultChunk(toolResult, channel);
                // log.info("tool result in executeReActLoopWithStreaming {}", toolResult);
                String toolResultString = extractToolResultContent(toolResult);

                // Create a proper MLTaskResponse for streaming instead of raw tool result
                MLTaskResponse toolChunk = createCompatibleToolChunk(toolResultString, sessionId, parentInteractionId);
                channel.sendResponseBatch(toolChunk);

                // After tool execution, use the exact same format as original:
                log
                    .info(
                        "action {}, actionInput {}, thought response {}",
                        finalAction,
                        finalActionInput,
                        currentModelOutput.get(THOUGHT_RESPONSE)
                    );

                // Continue to next iteration
                executeReActLoopWithStreaming(
                    llm,
                    mlAgent,
                    tools,
                    toolSpecMap,
                    tmpParameters,
                    interactions,
                    maxIterations,
                    tenantId,
                    listener,
                    functionCalling,
                    channel,
                    currentIteration + 1,
                    createToolExecutionOutput(finalAction, finalActionInput, toolResultString),
                    sessionId,
                    parentInteractionId
                );
            }, listener::onFailure), action, actionInput, toolParams, interactions, toolCallId, functionCalling);
        }
    }

    private Map<String, String> createToolExecutionOutput(String action, String actionInput, String toolResult) {
        Map<String, String> toolOutput = new HashMap<>();
        toolOutput.put(ACTION, action);
        toolOutput.put(ACTION_INPUT, actionInput);
        toolOutput.put("observation", toolResult);
        toolOutput.put(THOUGHT_RESPONSE, "Tool executed successfully");
        return toolOutput;
    }

    private MLTaskResponse createCompatibleToolChunk(String toolOutput, String sessionId, String parentInteractionId) {
        // Use ModelTensorOutput (ordinal 0) instead of unknown type
        // Map<String, Object> dataMap = Map.of("content", toolOutput, "is_last", false, "type", "tool_result");
        //
        // ModelTensor tensor = ModelTensor.builder().name("tool_response").dataAsMap(dataMap).build();
        //
        // ModelTensors tensors = ModelTensors.builder().mlModelTensors(List.of(tensor)).build();
        //
        // ModelTensorOutput output = ModelTensorOutput.builder().mlModelOutputs(List.of(tensors)).build();
        //
        // return new MLTaskResponse(output);
        List<ModelTensor> tensors = Arrays
            .asList(
                ModelTensor.builder().name("memory_id").result(sessionId).build(),
                ModelTensor.builder().name("parent_interaction_id").result(parentInteractionId).build(),
                ModelTensor
                    .builder()
                    .name("response")
                    .dataAsMap(Map.of("content", toolOutput, "is_last", false, "type", "tool_result"))
                    .build()
            );

        ModelTensors modelTensors = ModelTensors.builder().mlModelTensors(tensors).build();
        ModelTensorOutput output = ModelTensorOutput.builder().mlModelOutputs(List.of(modelTensors)).build();
        return new MLTaskResponse(output);
    }

    private String extractToolResultContent(Object toolResult) {
        if (toolResult instanceof String) {
            return (String) toolResult;
        }
        // Handle other possible types
        if (toolResult instanceof MLTaskResponse) {
            MLTaskResponse response = (MLTaskResponse) toolResult;
            return extractContent(response);
        }
        // Default fallback
        return toolResult.toString();
    }

    private MLInput buildMLInputFromParams(Map<String, String> params) {
        RemoteInferenceInputDataSet inputDataSet = RemoteInferenceInputDataSet.builder().parameters(params).build();

        return RemoteInferenceMLInput.builder().algorithm(FunctionName.REMOTE).inputDataset(inputDataSet).build();
    }

//    private String extractContent(Object chunk) {
//        try {
//            if (chunk instanceof MLTaskResponse) {
//                MLTaskResponse response = (MLTaskResponse) chunk;
//                ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
//                if (output != null && !output.getMlModelOutputs().isEmpty()) {
//                    ModelTensors tensors = output.getMlModelOutputs().get(0);
//                    if (!tensors.getMlModelTensors().isEmpty()) {
//                        Map<String, ?> dataMap = tensors.getMlModelTensors().get(0).getDataAsMap();
//                        if (dataMap.containsKey("content")) {
//                            return (String) dataMap.get("content");
//                        }
//                        if (dataMap.containsKey("response")) {
//                            return (String) dataMap.get("response");
//                        }
//                    }
//                }
//            }
//            return chunk.toString();
//        } catch (Exception e) {
//            log.error("Failed to extract content", e);
//            return "";
//        }
//    }

    private void runAgent(
        MLAgent mlAgent,
        Map<String, String> params,
        ActionListener<Object> listener,
        Memory memory,
        String sessionId,
        FunctionCalling functionCalling
    ) {
        List<MLToolSpec> toolSpecs = getMlToolSpecs(mlAgent, params);

        // Create a common method to handle both success and failure cases
        Consumer<List<MLToolSpec>> processTools = (allToolSpecs) -> {
            Map<String, Tool> tools = new HashMap<>();
            Map<String, MLToolSpec> toolSpecMap = new HashMap<>();
            createTools(toolFactories, params, allToolSpecs, tools, toolSpecMap, mlAgent);
            runReAct(mlAgent.getLlm(), tools, toolSpecMap, params, memory, sessionId, mlAgent.getTenantId(), listener, functionCalling);
        };

        // Fetch MCP tools and handle both success and failure cases
        getMcpToolSpecs(mlAgent, client, sdkClient, encryptor, ActionListener.wrap(mcpTools -> {
            toolSpecs.addAll(mcpTools);
            processTools.accept(toolSpecs);
        }, e -> {
            log.error("Failed to get MCP tools, continuing with base tools only", e);
            processTools.accept(toolSpecs);
        }));
    }

    private void runReAct(
        LLMSpec llm,
        Map<String, Tool> tools,
        Map<String, MLToolSpec> toolSpecMap,
        Map<String, String> parameters,
        Memory memory,
        String sessionId,
        String tenantId,
        ActionListener<Object> listener,
        FunctionCalling functionCalling
    ) {
        Map<String, String> tmpParameters = constructLLMParams(llm, parameters);
        String prompt = constructLLMPrompt(tools, tmpParameters);
        tmpParameters.put(PROMPT, prompt);
        final String finalPrompt = prompt;

        String question = tmpParameters.get(MLAgentExecutor.QUESTION);
        String parentInteractionId = tmpParameters.get(MLAgentExecutor.PARENT_INTERACTION_ID);
        boolean verbose = Boolean.parseBoolean(tmpParameters.getOrDefault(VERBOSE, "false"));
        boolean traceDisabled = tmpParameters.containsKey(DISABLE_TRACE) && Boolean.parseBoolean(tmpParameters.get(DISABLE_TRACE));

        // Create root interaction.
        ConversationIndexMemory conversationIndexMemory = (ConversationIndexMemory) memory;

        // Trace number
        AtomicInteger traceNumber = new AtomicInteger(0);

        AtomicReference<StepListener<MLTaskResponse>> lastLlmListener = new AtomicReference<>();
        AtomicReference<String> lastThought = new AtomicReference<>();
        AtomicReference<String> lastAction = new AtomicReference<>();
        AtomicReference<String> lastActionInput = new AtomicReference<>();
        AtomicReference<String> lastToolSelectionResponse = new AtomicReference<>();
        Map<String, Object> additionalInfo = new ConcurrentHashMap<>();

        StepListener firstListener = new StepListener<MLTaskResponse>();
        lastLlmListener.set(firstListener);
        StepListener<?> lastStepListener = firstListener;

        StringBuilder scratchpadBuilder = new StringBuilder();
        List<String> interactions = new CopyOnWriteArrayList<>();

        StringSubstitutor tmpSubstitutor = new StringSubstitutor(Map.of(SCRATCHPAD, scratchpadBuilder.toString()), "${parameters.", "}");
        AtomicReference<String> newPrompt = new AtomicReference<>(tmpSubstitutor.replace(prompt));
        tmpParameters.put(PROMPT, newPrompt.get());

        List<ModelTensors> traceTensors = createModelTensors(sessionId, parentInteractionId);
        int maxIterations = Integer.parseInt(tmpParameters.getOrDefault(MAX_ITERATION, DEFAULT_MAX_ITERATIONS));
        for (int i = 0; i < maxIterations; i++) {
            int finalI = i;
            StepListener<?> nextStepListener = new StepListener<>();

            lastStepListener.whenComplete(output -> {
                StringBuilder sessionMsgAnswerBuilder = new StringBuilder();
                if (finalI % 2 == 0) {
                    MLTaskResponse llmResponse = (MLTaskResponse) output;
                    ModelTensorOutput tmpModelTensorOutput = (ModelTensorOutput) llmResponse.getOutput();
                    List<String> llmResponsePatterns = gson.fromJson(tmpParameters.get("llm_response_pattern"), List.class);
                    Map<String, String> modelOutput = parseLLMOutput(
                        parameters,
                        tmpModelTensorOutput,
                        llmResponsePatterns,
                        tools.keySet(),
                        interactions,
                        functionCalling
                    );

                    String thought = String.valueOf(modelOutput.get(THOUGHT));
                    String toolCallId = String.valueOf(modelOutput.get("tool_call_id"));
                    String action = String.valueOf(modelOutput.get(ACTION));
                    String actionInput = String.valueOf(modelOutput.get(ACTION_INPUT));
                    String thoughtResponse = modelOutput.get(THOUGHT_RESPONSE);
                    String finalAnswer = modelOutput.get(FINAL_ANSWER);

                    if (finalAnswer != null) {
                        finalAnswer = finalAnswer.trim();
                        sendFinalAnswer(
                            sessionId,
                            listener,
                            question,
                            parentInteractionId,
                            verbose,
                            traceDisabled,
                            traceTensors,
                            conversationIndexMemory,
                            traceNumber,
                            additionalInfo,
                            finalAnswer
                        );
                        cleanUpResource(tools);
                        return;
                    }

                    sessionMsgAnswerBuilder.append(thought);
                    lastThought.set(thought);
                    lastAction.set(action);
                    lastActionInput.set(actionInput);
                    lastToolSelectionResponse.set(thoughtResponse);

                    traceTensors
                        .add(
                            ModelTensors
                                .builder()
                                .mlModelTensors(List.of(ModelTensor.builder().name("response").result(thoughtResponse).build()))
                                .build()
                        );

                    saveTraceData(
                        conversationIndexMemory,
                        memory.getType(),
                        question,
                        thoughtResponse,
                        sessionId,
                        traceDisabled,
                        parentInteractionId,
                        traceNumber,
                        "LLM"
                    );

                    if (tools.containsKey(action)) {
                        Map<String, String> toolParams = constructToolParams(
                            tools,
                            toolSpecMap,
                            question,
                            lastActionInput,
                            action,
                            actionInput
                        );
                        runTool(
                            tools,
                            toolSpecMap,
                            tmpParameters,
                            (ActionListener<Object>) nextStepListener,
                            action,
                            actionInput,
                            toolParams,
                            interactions,
                            toolCallId,
                            functionCalling
                        );
                    } else {
                        String res = String.format(Locale.ROOT, "Failed to run the tool %s which is unsupported.", action);
                        StringSubstitutor substitutor = new StringSubstitutor(
                            Map.of(SCRATCHPAD, scratchpadBuilder.toString()),
                            "${parameters.",
                            "}"
                        );
                        newPrompt.set(substitutor.replace(finalPrompt));
                        tmpParameters.put(PROMPT, newPrompt.get());
                        ((ActionListener<Object>) nextStepListener).onResponse(res);
                    }
                } else {
                    addToolOutputToAddtionalInfo(toolSpecMap, lastAction, additionalInfo, output);

                    String toolResponse = constructToolResponse(
                        tmpParameters,
                        lastAction,
                        lastActionInput,
                        lastToolSelectionResponse,
                        output
                    );
                    scratchpadBuilder.append(toolResponse).append("\n\n");

                    saveTraceData(
                        conversationIndexMemory,
                        "ReAct",
                        lastActionInput.get(),
                        outputToOutputString(output),
                        sessionId,
                        traceDisabled,
                        parentInteractionId,
                        traceNumber,
                        lastAction.get()
                    );

                    StringSubstitutor substitutor = new StringSubstitutor(Map.of(SCRATCHPAD, scratchpadBuilder), "${parameters.", "}");
                    newPrompt.set(substitutor.replace(finalPrompt));
                    tmpParameters.put(PROMPT, newPrompt.get());
                    if (interactions.size() > 0) {
                        tmpParameters.put(INTERACTIONS, ", " + String.join(", ", interactions));
                    }

                    sessionMsgAnswerBuilder.append(outputToOutputString(output));
                    traceTensors
                        .add(
                            ModelTensors
                                .builder()
                                .mlModelTensors(
                                    Collections
                                        .singletonList(
                                            ModelTensor.builder().name("response").result(sessionMsgAnswerBuilder.toString()).build()
                                        )
                                )
                                .build()
                        );

                    if (finalI == maxIterations - 1) {
                        if (verbose) {
                            listener.onResponse(ModelTensorOutput.builder().mlModelOutputs(traceTensors).build());
                        } else {
                            List<ModelTensors> finalModelTensors = createFinalAnswerTensors(
                                createModelTensors(sessionId, parentInteractionId),
                                List.of(ModelTensor.builder().name("response").dataAsMap(Map.of("response", lastThought.get())).build())
                            );
                            listener.onResponse(ModelTensorOutput.builder().mlModelOutputs(finalModelTensors).build());
                        }
                    } else {
                        ActionRequest request = new MLPredictionTaskRequest(
                            llm.getModelId(),
                            RemoteInferenceMLInput
                                .builder()
                                .algorithm(FunctionName.REMOTE)
                                .inputDataset(RemoteInferenceInputDataSet.builder().parameters(tmpParameters).build())
                                .build(),
                            null,
                            tenantId
                        );
                        client.execute(MLPredictionTaskAction.INSTANCE, request, (ActionListener<MLTaskResponse>) nextStepListener);
                    }
                }
            }, e -> {
                log.error("Failed to run chat agent", e);
                listener.onFailure(e);
            });
            if (i < maxIterations - 1) {
                lastStepListener = nextStepListener;
            }
        }

        ActionRequest request = new MLPredictionTaskRequest(
            llm.getModelId(),
            RemoteInferenceMLInput
                .builder()
                .algorithm(FunctionName.REMOTE)
                .inputDataset(RemoteInferenceInputDataSet.builder().parameters(tmpParameters).build())
                .build(),
            null,
            tenantId
        );
        client.execute(MLPredictionTaskAction.INSTANCE, request, firstListener);
    }

    private static List<ModelTensors> createFinalAnswerTensors(List<ModelTensors> sessionId, List<ModelTensor> lastThought) {
        List<ModelTensors> finalModelTensors = sessionId;
        finalModelTensors.add(ModelTensors.builder().mlModelTensors(lastThought).build());
        return finalModelTensors;
    }

    private static String constructToolResponse(
        Map<String, String> tmpParameters,
        AtomicReference<String> lastAction,
        AtomicReference<String> lastActionInput,
        AtomicReference<String> lastToolSelectionResponse,
        Object output
    ) throws PrivilegedActionException {
        String toolResponse = tmpParameters.get(TOOL_RESPONSE);
        StringSubstitutor toolResponseSubstitutor = new StringSubstitutor(
            Map
                .of(
                    "llm_tool_selection_response",
                    lastToolSelectionResponse.get(),
                    "tool_name",
                    lastAction.get(),
                    "tool_input",
                    lastActionInput.get(),
                    "observation",
                    outputToOutputString(output)
                ),
            "${parameters.",
            "}"
        );
        toolResponse = toolResponseSubstitutor.replace(toolResponse);
        return toolResponse;
    }

    private static void addToolOutputToAddtionalInfo(
        Map<String, MLToolSpec> toolSpecMap,
        AtomicReference<String> lastAction,
        Map<String, Object> additionalInfo,
        Object output
    ) throws PrivilegedActionException {
        MLToolSpec toolSpec = toolSpecMap.get(lastAction.get());
        if (toolSpec != null && toolSpec.isIncludeOutputInAgentResponse()) {
            String outputString = outputToOutputString(output);
            String toolOutputKey = String.format("%s.output", getToolName(toolSpec));
            if (additionalInfo.get(toolOutputKey) != null) {
                List<String> list = (List<String>) additionalInfo.get(toolOutputKey);
                list.add(outputString);
            } else {
                additionalInfo.put(toolOutputKey, Lists.newArrayList(outputString));
            }
        }
    }

    private static void runTool(
        Map<String, Tool> tools,
        Map<String, MLToolSpec> toolSpecMap,
        Map<String, String> tmpParameters,
        ActionListener<Object> nextStepListener,
        String action,
        String actionInput,
        Map<String, String> toolParams,
        List<String> interactions,
        String toolCallId,
        FunctionCalling functionCalling
    ) {
        if (tools.get(action).validate(toolParams)) {
            try {
                String finalAction = action;
                ActionListener<Object> toolListener = ActionListener.wrap(r -> {
                    if (functionCalling != null) {
                        String outputResponse = parseResponse(filterToolOutput(toolParams, r));
                        List<Map<String, Object>> toolResults = List
                            .of(Map.of(TOOL_CALL_ID, toolCallId, TOOL_RESULT, Map.of("text", outputResponse)));
                        List<LLMMessage> llmMessages = functionCalling.supply(toolResults);
                        // TODO: support multiple tool calls at the same time so that multiple LLMMessages can be generated here
                        interactions.add(llmMessages.getFirst().getResponse());
                    } else {
                        interactions
                            .add(
                                substitute(
                                    tmpParameters.get(INTERACTION_TEMPLATE_TOOL_RESPONSE),
                                    Map.of(TOOL_CALL_ID, toolCallId, "tool_response", processTextDoc(StringUtils.toJson(r))),
                                    INTERACTIONS_PREFIX
                                )
                            );
                    }
                    nextStepListener.onResponse(r);
                }, e -> {
                    interactions
                        .add(
                            substitute(
                                tmpParameters.get(INTERACTION_TEMPLATE_TOOL_RESPONSE),
                                Map.of(TOOL_CALL_ID, toolCallId, "tool_response", "Tool " + action + " failed: " + e.getMessage()),
                                INTERACTIONS_PREFIX
                            )
                        );
                    nextStepListener
                        .onResponse(
                            String
                                .format(
                                    Locale.ROOT,
                                    "Failed to run the tool %s with the error message %s.",
                                    finalAction,
                                    e.getMessage().replaceAll("\\n", "\n")
                                )
                        );
                });
                if (tools.get(action) instanceof MLModelTool) {
                    Map<String, String> llmToolTmpParameters = new HashMap<>();
                    llmToolTmpParameters.putAll(tmpParameters);
                    llmToolTmpParameters.putAll(toolSpecMap.get(action).getParameters());
                    llmToolTmpParameters.put(MLAgentExecutor.QUESTION, actionInput);
                    tools.get(action).run(llmToolTmpParameters, toolListener); // run tool
                } else {
                    Map<String, String> parameters = new HashMap<>();
                    parameters.putAll(tmpParameters);
                    parameters.putAll(toolParams);
                    tools.get(action).run(parameters, toolListener); // run tool
                }
            } catch (Exception e) {
                log.error("Failed to run tool {}", action, e);
                nextStepListener
                    .onResponse(String.format(Locale.ROOT, "Failed to run the tool %s with the error message %s.", action, e.getMessage()));
            }
        } else { // TODO: add failure to interaction to let LLM regenerate ?
            String res = String.format(Locale.ROOT, "Failed to run the tool %s due to wrong input %s.", action, actionInput);
            nextStepListener.onResponse(res);
        }
    }

    public static void saveTraceData(
        ConversationIndexMemory conversationIndexMemory,
        String memory,
        String question,
        String thoughtResponse,
        String sessionId,
        boolean traceDisabled,
        String parentInteractionId,
        AtomicInteger traceNumber,
        String origin
    ) {
        if (conversationIndexMemory != null) {
            ConversationIndexMessage msgTemp = ConversationIndexMessage
                .conversationIndexMessageBuilder()
                .type(memory)
                .question(question)
                .response(thoughtResponse)
                .finalAnswer(false)
                .sessionId(sessionId)
                .build();
            if (!traceDisabled) {
                conversationIndexMemory.save(msgTemp, parentInteractionId, traceNumber.addAndGet(1), origin);
            }
        }
    }

    private void saveToMemoryOnly(
            String sessionId,
            ActionListener<Object> listener,
            String question,
            String parentInteractionId,
            boolean verbose,
            boolean traceDisabled,
            List<ModelTensors> cotModelTensors,
            ConversationIndexMemory conversationIndexMemory,
            AtomicInteger traceNumber,
            Map<String, Object> additionalInfo,
            String finalAnswer,
            TransportChannel channel
    ) {
        // Mark agent as completed
        agentCompleted = true;

        // Send final completion chunk to close stream
        MLTaskResponse completionChunk = createCompletionChunk(sessionId, parentInteractionId);
        try {
            channel.sendResponseBatch(completionChunk);
        } catch (Exception e) {
            log.warn("Failed to send completion chunk: {}", e.getMessage());
        }

        if (conversationIndexMemory != null) {
            String copyOfFinalAnswer = finalAnswer;
            ActionListener saveTraceListener = ActionListener.wrap(r -> {
                conversationIndexMemory
                        .getMemoryManager()
                        .updateInteraction(
                                parentInteractionId,
                                Map.of(AI_RESPONSE_FIELD, copyOfFinalAnswer, ADDITIONAL_INFO_FIELD, additionalInfo),
                                ActionListener.wrap(res -> {
                                    // Don't send response - streaming already handled it
                                    listener.onResponse("Streaming completed");
                                }, e -> { listener.onFailure(e); })
                        );
            }, e -> { listener.onFailure(e); });
            saveMessage(
                    conversationIndexMemory,
                    question,
                    finalAnswer,
                    sessionId,
                    parentInteractionId,
                    traceNumber,
                    true,
                    traceDisabled,
                    saveTraceListener
            );
        } else {
            listener.onResponse("Streaming completed");
        }
    }

    private MLTaskResponse createCompletionChunk(String sessionId, String parentInteractionId) {
        List<ModelTensor> tensors = Arrays.asList(
                ModelTensor.builder().name("memory_id").result(sessionId).build(),
                ModelTensor.builder().name("parent_interaction_id").result(parentInteractionId).build(),
                ModelTensor.builder()
                        .name("response")
                        .dataAsMap(Map.of(
                                "content", "",
                                "is_last", true
                        ))
                        .build()
        );

        ModelTensors modelTensors = ModelTensors.builder().mlModelTensors(tensors).build();
        ModelTensorOutput output = ModelTensorOutput.builder().mlModelOutputs(List.of(modelTensors)).build();
        return new MLTaskResponse(output);
    }


    private void sendFinalAnswer(
        String sessionId,
        ActionListener<Object> listener,
        String question,
        String parentInteractionId,
        boolean verbose,
        boolean traceDisabled,
        List<ModelTensors> cotModelTensors,
        ConversationIndexMemory conversationIndexMemory,
        AtomicInteger traceNumber,
        Map<String, Object> additionalInfo,
        String finalAnswer
    ) {
        if (conversationIndexMemory != null) {
            String copyOfFinalAnswer = finalAnswer;
            ActionListener saveTraceListener = ActionListener.wrap(r -> {
                conversationIndexMemory
                    .getMemoryManager()
                    .updateInteraction(
                        parentInteractionId,
                        Map.of(AI_RESPONSE_FIELD, copyOfFinalAnswer, ADDITIONAL_INFO_FIELD, additionalInfo),
                        ActionListener.wrap(res -> {
                            returnFinalResponse(
                                sessionId,
                                listener,
                                parentInteractionId,
                                verbose,
                                cotModelTensors,
                                additionalInfo,
                                copyOfFinalAnswer
                            );
                        }, e -> { listener.onFailure(e); })
                    );
            }, e -> { listener.onFailure(e); });
            saveMessage(
                conversationIndexMemory,
                question,
                finalAnswer,
                sessionId,
                parentInteractionId,
                traceNumber,
                true,
                traceDisabled,
                saveTraceListener
            );
        } else {
            returnFinalResponse(sessionId, listener, parentInteractionId, verbose, cotModelTensors, additionalInfo, finalAnswer);
        }
    }

    public static List<ModelTensors> createModelTensors(String sessionId, String parentInteractionId) {
        List<ModelTensors> cotModelTensors = new ArrayList<>();

        cotModelTensors
            .add(
                ModelTensors
                    .builder()
                    .mlModelTensors(
                        List
                            .of(
                                ModelTensor.builder().name(MLAgentExecutor.MEMORY_ID).result(sessionId).build(),
                                ModelTensor.builder().name(MLAgentExecutor.PARENT_INTERACTION_ID).result(parentInteractionId).build()
                            )
                    )
                    .build()
            );
        return cotModelTensors;
    }

    private static String constructLLMPrompt(Map<String, Tool> tools, Map<String, String> tmpParameters) {
        String prompt = tmpParameters.getOrDefault(PROMPT, PromptTemplate.PROMPT_TEMPLATE);
        StringSubstitutor promptSubstitutor = new StringSubstitutor(tmpParameters, "${parameters.", "}");
        prompt = promptSubstitutor.replace(prompt);
        prompt = AgentUtils.addPrefixSuffixToPrompt(tmpParameters, prompt);
        prompt = AgentUtils.addToolsToPrompt(tools, tmpParameters, getToolNames(tools), prompt);
        prompt = AgentUtils.addIndicesToPrompt(tmpParameters, prompt);
        prompt = AgentUtils.addExamplesToPrompt(tmpParameters, prompt);
        prompt = AgentUtils.addChatHistoryToPrompt(tmpParameters, prompt);
        prompt = AgentUtils.addContextToPrompt(tmpParameters, prompt);
        return prompt;
    }

    private static Map<String, String> constructLLMParams(LLMSpec llm, Map<String, String> parameters) {
        Map<String, String> tmpParameters = new HashMap<>();
        if (llm.getParameters() != null) {
            tmpParameters.putAll(llm.getParameters());
        }
        tmpParameters.putAll(parameters);
        if (!tmpParameters.containsKey("stop")) {
            tmpParameters.put("stop", gson.toJson(new String[] { "\nObservation:", "\n\tObservation:" }));
        }
        if (!tmpParameters.containsKey("stop_sequences")) {
            tmpParameters
                .put(
                    "stop_sequences",
                    gson
                        .toJson(
                            new String[] {
                                "\n\nHuman:",
                                "\nObservation:",
                                "\n\tObservation:",
                                "\nObservation",
                                "\n\tObservation",
                                "\n\nQuestion" }
                        )
                );
        }

        tmpParameters.putIfAbsent(PROMPT_PREFIX, PromptTemplate.PROMPT_TEMPLATE_PREFIX);
        tmpParameters.putIfAbsent(PROMPT_SUFFIX, PromptTemplate.PROMPT_TEMPLATE_SUFFIX);
        tmpParameters.putIfAbsent(RESPONSE_FORMAT_INSTRUCTION, PromptTemplate.PROMPT_FORMAT_INSTRUCTION);
        tmpParameters.putIfAbsent(TOOL_RESPONSE, PromptTemplate.PROMPT_TEMPLATE_TOOL_RESPONSE);
        return tmpParameters;
    }

    private static void returnFinalResponse(
        String sessionId,
        ActionListener<Object> listener,
        String parentInteractionId,
        boolean verbose,
        List<ModelTensors> cotModelTensors, // AtomicBoolean getFinalAnswer,
        Map<String, Object> additionalInfo,
        String finalAnswer2
    ) {
        cotModelTensors
            .add(
                ModelTensors.builder().mlModelTensors(List.of(ModelTensor.builder().name("response").result(finalAnswer2).build())).build()
            );

        List<ModelTensors> finalModelTensors = createFinalAnswerTensors(
            createModelTensors(sessionId, parentInteractionId),
            List
                .of(
                    ModelTensor
                        .builder()
                        .name("response")
                        .dataAsMap(ImmutableMap.of("response", finalAnswer2, ADDITIONAL_INFO_FIELD, additionalInfo))
                        .build()
                )
        );
        if (verbose) {
            listener.onResponse(ModelTensorOutput.builder().mlModelOutputs(cotModelTensors).build());
        } else {
            listener.onResponse(ModelTensorOutput.builder().mlModelOutputs(finalModelTensors).build());
        }
    }

    private void saveMessage(
        ConversationIndexMemory memory,
        String question,
        String finalAnswer,
        String sessionId,
        String parentInteractionId,
        AtomicInteger traceNumber,
        boolean isFinalAnswer,
        boolean traceDisabled,
        ActionListener listener
    ) {
        ConversationIndexMessage msgTemp = ConversationIndexMessage
            .conversationIndexMessageBuilder()
            .type(memory.getType())
            .question(question)
            .response(finalAnswer)
            .finalAnswer(isFinalAnswer)
            .sessionId(sessionId)
            .build();
        if (traceDisabled) {
            listener.onResponse(true);
        } else {
            memory.save(msgTemp, parentInteractionId, traceNumber.addAndGet(1), "LLM", listener);
        }
    }
}
