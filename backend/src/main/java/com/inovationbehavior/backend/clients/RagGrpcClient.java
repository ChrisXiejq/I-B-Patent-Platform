package com.inovationbehavior.backend.clients;
import com.inovationbehavior.backend.Protos.RagRequest;
import com.inovationbehavior.backend.Protos.RagResponse;
import com.inovationbehavior.backend.Protos.RagServiceGrpc;
import lombok.extern.slf4j.Slf4j;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class RagGrpcClient {

    @GrpcClient("rag-service")
    private RagServiceGrpc.RagServiceBlockingStub ragStub;

    public String getRagAnswer(String userQuery) {

        RagRequest request = RagRequest.newBuilder()
                .setUserQuery(userQuery)
                .build();

        RagResponse response;
        try {
            log.info("Calling rag service");
            response = ragStub.getRagAnswer(request);
        } catch (Exception e) {
            throw new RuntimeException("调用 RAG gRPC 服务失败", e);
        }

        if (response.getAnswer().isEmpty()) {
            throw new RuntimeException(
                    "RAG 服务返回失败: " + response.getAnswer()
            );
        }

        return response.getAnswer();
    }
}