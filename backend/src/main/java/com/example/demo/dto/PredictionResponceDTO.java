package com.example.demo.dto;

import com.example.demo.model.EthTransaction;
import lombok.Data;

import java.sql.Timestamp;
import java.util.List;

@Data
public class PredictionResponceDTO {
    private String address;
    private Prediction prediction;
    private List<EthTransaction> transactions_used;
    @Data
    public static class Prediction {
        private int total_transactions;
        private String timestamp;
        private String prediction;
        private double fraud_probability;
        private double normal_probability;
        private double confidence;
        private Boolean is_fraud;
        private int features_used;
        private int sequence_length;
    }
}
