package com.example.demo.controllers;

import com.example.demo.dto.PredictionResponceDTO;
import com.example.demo.services.ExplanationService;
import com.example.demo.services.PredictionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class AccountAnalyserController {
    private final ExplanationService explanationService;
    private final PredictionService predictionService;


    public AccountAnalyserController(ExplanationService explanationService, PredictionService predictionService) {
        this.explanationService = explanationService;
        this.predictionService = predictionService;
    }

    @PostMapping("/analyse")
    public ResponseEntity analyse(@RequestParam String address) {
        try{
            PredictionResponceDTO response = predictionService.getPrediction(address);
            return  ResponseEntity.ok(
                    explanationService.ExplainFraud(response)
            );
        }catch (Exception e){
            return  ResponseEntity.badRequest().body(e.getMessage());
        }
    }


}
