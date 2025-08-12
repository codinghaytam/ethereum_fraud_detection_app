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
    @Mapping(source = "predictionResponseDTO.prediction", target = "prediction")
    @Mapping(source = "predictionResponseDTO.address", target = "address")
    @Mapping(source = "predictionResponseDTO.confidence", target = "confidence")
    @Mapping(source = "predictionResponseDTO.total_transactions", target = "total_transactions")
    @Mapping(source = "predictionResponseDTO.fraud_probability", target = "fraud_probability")
    PredictionAndExplanationDTO toPredictionAndExplanationDTO(PredictionResponceDTO predictionResponseDTO, List<Raison> explanations);
}
