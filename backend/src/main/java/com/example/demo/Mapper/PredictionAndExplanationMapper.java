package com.example.demo.Mapper;

import com.example.demo.dto.PredictionAndExplanationDTO;
import com.example.demo.dto.PredictionResponceDTO;
import com.example.demo.model.Raison;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.ReportingPolicy;
import org.springframework.context.annotation.Bean;

import java.util.List;
@Mapper(componentModel = "spring")

public interface PredictionAndExplanationMapper {
    @Mapping(source = "explanations", target = "explanations")
    @Mapping(source = "predictionResponseDTO.prediction.prediction", target = "prediction")
    @Mapping(source = "predictionResponseDTO.prediction.is_fraud", target = "is_fraud")
    @Mapping(source = "predictionResponseDTO.address", target = "address")
    @Mapping(source = "predictionResponseDTO.prediction.confidence", target = "confidence")
    @Mapping(source = "predictionResponseDTO.prediction.total_transactions", target = "total_transactions")
    @Mapping(source = "predictionResponseDTO.prediction.timestamp", target = "analysis_timestamp")
    @Mapping(source = "predictionResponseDTO.prediction.fraud_probability", target = "fraud_probability")
    PredictionAndExplanationDTO toPredictionAndExplanationDTO(PredictionResponceDTO predictionResponseDTO, List<Raison> explanations);
}
