# Spring Boot Backend Dockerfile
FROM maven:3.9.6-eclipse-temurin-17 as builder

WORKDIR /app

# Accept build arguments
ARG SPRING_PROFILES_ACTIVE=docker

# Copy pom.xml
COPY pom.xml .

# Download dependencies
RUN mvn dependency:go-offline -B

# Copy source code
COPY src ./src

# Build the application
RUN mvn clean package -DskipTests -Dspring.profiles.active=$SPRING_PROFILES_ACTIVE

# Production stage
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

# Copy the built jar
COPY --from=builder /app/target/*.jar app.jar

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

USER appuser

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar", "--spring.profiles.active=docker"]
