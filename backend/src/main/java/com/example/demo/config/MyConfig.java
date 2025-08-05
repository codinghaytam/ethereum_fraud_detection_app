package com.example.demo.config;
import com.example.demo.model.Raison;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.web.client.RestTemplate;

import java.util.List;


@Configuration
public class MyConfig {
    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder){
        return builder.build();
    }

    @Bean
    public BeanOutputConverter<List<Raison>> raisonListOutputConverter(){
        return new BeanOutputConverter(new ParameterizedTypeReference<List<Raison>>() {
        });
    }

}
