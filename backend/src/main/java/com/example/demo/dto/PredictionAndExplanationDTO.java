package com.example.demo.dto;

import com.example.demo.model.Raison;
import lombok.Data;

import java.sql.Timestamp;
import java.util.List;

@Data
public class PredictionAndExplanationDTO {
    private String address;
    private String prediction;
    private Double confidence;
    private Double fraud_probability;
    private Integer total_transactions;
    private List<Raison> explanations;

}
