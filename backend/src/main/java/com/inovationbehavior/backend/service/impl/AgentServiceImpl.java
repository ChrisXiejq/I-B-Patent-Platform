package com.inovationbehavior.backend.service.impl;

import com.inovationbehavior.backend.clients.AgentGrpcClient;
import com.inovationbehavior.backend.service.intf.AgentService;
import org.springframework.beans.factory.annotation.Autowired;

import org.springframework.stereotype.Service;

@Service
public class AgentServiceImpl implements AgentService {
    @Autowired
    private AgentGrpcClient agentGrpcClient;
    @Override
    public String chat(String userQuery) {
        String fullContext = userQuery;
        return agentGrpcClient.chat(fullContext);
    }
}
