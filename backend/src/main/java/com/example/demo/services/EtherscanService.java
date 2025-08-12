package com.example.demo.services;

import com.example.demo.model.EthTransaction;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.util.List;

@Service
public class EtherscanService {
    private final ObjectMapper objectMapper;
    private final RestTemplate restTemplate;
    private final String uriEtherscan = "https://api.etherscan.io/api";
    @Value("${etherscan.key}")
    private String etherscanKey;

    public EtherscanService(ObjectMapper objectMapper, RestTemplate restTemplate) {
        this.objectMapper = objectMapper;
        this.restTemplate = restTemplate;
    }


}

