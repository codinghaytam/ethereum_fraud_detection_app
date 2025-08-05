package com.example.demo.services;

import com.example.demo.dto.PredictionResponceDTO;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class PredictionService {
    private final RestTemplate restTemplate;
    @Value("${prediction.api.uri}")
    private String uriPrediction;
    private final ObjectMapper objectMapper;
    public PredictionService(RestTemplate restTemplate, ObjectMapper objectMapper) {
        this.restTemplate = restTemplate;
        this.objectMapper = objectMapper;
    }

    public PredictionResponceDTO getPrediction(String address) {
        String url = uriPrediction + "?address=" + address;
        JsonNode response = restTemplate.postForObject(url, objectMapper, JsonNode.class);
        if (response == null ) {
            throw new RuntimeException("Invalid response from prediction API");
        }
        try {
            return objectMapper.treeToValue(response, PredictionResponceDTO.class);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Error parsing prediction response: " + e.getMessage(), e);
        }
    }

}
