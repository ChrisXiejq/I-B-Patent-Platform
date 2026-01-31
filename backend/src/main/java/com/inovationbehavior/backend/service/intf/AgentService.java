package com.inovationbehavior.backend.service.intf;

public interface AgentService {
    /**
     * 与智能体服务交互，获取响应
     *
     * @param userQuery 用户查询
     * @return 智能体服务的响应
     */
    String chat(String userQuery);
}
