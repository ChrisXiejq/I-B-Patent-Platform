package com.inovationbehavior.backend.clients;
import com.inovationbehavior.backend.Protos.AgentRequest;
import com.inovationbehavior.backend.Protos.AgentResponse;
import com.inovationbehavior.backend.Protos.AgentServiceGrpc;
import lombok.extern.slf4j.Slf4j;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class AgentGrpcClient {

    @GrpcClient("agent-service")
    private AgentServiceGrpc.AgentServiceBlockingStub agentStub;

    public String chat(String userQuery) {

        AgentRequest request = AgentRequest.newBuilder()
                .setQuery(userQuery)
                .build();

        AgentResponse response;
        try {
            log.info("Calling agent service");
            response = agentStub.chat(request);
        } catch (Exception e) {
            throw new RuntimeException("调用 Agent gRPC 服务失败", e);
        }

        if (response.getAnswer().isEmpty()) {
            throw new RuntimeException(
                    "Agent 服务返回失败: " + response.getAnswer()
            );
        }

        return response.getAnswer();
    }
}