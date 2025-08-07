package com.example.demo.services;

import com.example.demo.Mapper.PredictionAndExplanationMapper;
import com.example.demo.dto.PredictionAndExplanationDTO;
import com.example.demo.dto.PredictionResponceDTO;
import com.example.demo.model.Raison;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.ai.converter.ListOutputConverter;
import org.springframework.ai.mistralai.MistralAiChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ExplanationService {
    private final MistralAiChatModel mistralAiChatModel;
    private final PredictionAndExplanationMapper predictionAndExplanationMapper;
    private final BeanOutputConverter<List<Raison>> raisonBeanOutputConverter;
    private final ObjectMapper objectMapper;
    public ExplanationService(MistralAiChatModel mistralAiChatModel, PredictionAndExplanationMapper predictionAndExplanationMapper, BeanOutputConverter<List<Raison>> raisonBeanOutputConverter, ObjectMapper objectMapper) {
        this.mistralAiChatModel = mistralAiChatModel;
        this.predictionAndExplanationMapper = predictionAndExplanationMapper;
        this.raisonBeanOutputConverter = raisonBeanOutputConverter;
        this.objectMapper = objectMapper;
    }

    public PredictionAndExplanationDTO ExplainFraud(PredictionResponceDTO addressPrediction) throws JsonProcessingException {
        String transactions = addressPrediction.getTransactions_used().toString();
        String aiResponce = mistralAiChatModel.call("you are a blockchain expert ,Analyse why the following address is fraudulant: " + addressPrediction.getAddress() + " here are the transactions: " + transactions + "give 3 probable raisons with examples of popular scams including the raison"
        + " the response format:" + raisonBeanOutputConverter.getFormat());
        if (aiResponce == null || aiResponce.isEmpty()) {
            throw new JsonProcessingException("AI response is empty or null") {};
        }
        System.out.println("AI Response: " + aiResponce);
        List<Raison> explanations = raisonBeanOutputConverter.convert(aiResponce);
        return predictionAndExplanationMapper.toPredictionAndExplanationDTO(addressPrediction, explanations);
    }

}