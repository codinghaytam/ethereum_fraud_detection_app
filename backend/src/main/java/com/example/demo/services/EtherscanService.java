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

    public List<EthTransaction> getEthTransactions(String address) throws JsonProcessingException {
        UriComponentsBuilder builder = UriComponentsBuilder.fromUriString(uriEtherscan)
                .queryParam("module", "account")
                .queryParam("action", "txlist")
                .queryParam("address", address)
                .queryParam("apikey", etherscanKey)
                .queryParam("page",1)
                .queryParam("startblock", 0)
                .queryParam("endblock", 999999999)
                .queryParam("offset", 100)
                .queryParam("sort", "asc");

        JsonNode result = objectMapper.readTree(restTemplate.getForObject(builder.toUriString(), String.class));
        if (!result.has("result") || !result.get("result").isArray()) {
            throw new JsonProcessingException("Invalid response from Etherscan API: " + result.toString()) {};
        }else{
        }
        return objectMapper.readValue(
                result.get("result").toString(),
                objectMapper.getTypeFactory().constructCollectionType(List.class, EthTransaction.class)
        );
    }
}
