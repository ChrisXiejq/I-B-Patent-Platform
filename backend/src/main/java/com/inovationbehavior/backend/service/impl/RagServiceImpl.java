package com.inovationbehavior.backend.service.impl;

import com.inovationbehavior.backend.clients.RagGrpcClient;
import com.inovationbehavior.backend.service.intf.RagService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class RagServiceImpl implements RagService {
    @Autowired
    private RagGrpcClient ragGrpcClient;
    @Override
    public String getRagAnswer(String userQuery, String patentNo) {
        String fullContext = userQuery + "and patent no is " + patentNo;
        return ragGrpcClient.getRagAnswer(fullContext);
    }
}
