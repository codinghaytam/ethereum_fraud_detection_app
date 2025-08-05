package com.example.demo.dto;

import com.example.demo.model.Raison;
import lombok.Data;

import java.sql.Timestamp;
import java.util.List;

@Data
public class PredictionAndExplanationDTO {
    private String prediction;
    private Boolean is_fraud;
    private String address;
    private int total_transactions;
    private String analysis_timestamp;
    private double confidence;
    private List<Raison> explanations;

}
