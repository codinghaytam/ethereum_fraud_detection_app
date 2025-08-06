# Frontend Dockerfile
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files first
COPY package*.json ./

# Install dependencies
RUN npm i

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:1.25-alpine

# Copy built app to nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# Use exec form to avoid shell issues
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
