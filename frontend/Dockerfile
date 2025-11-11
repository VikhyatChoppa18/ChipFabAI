# Multi-stage build for cost optimization - smaller image size
FROM node:18-alpine AS build

WORKDIR /app

# Accept build argument for API URL
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=$REACT_APP_API_URL

# Copy package files first (for better Docker layer caching)
COPY package*.json ./

# Install all dependencies (including dev dependencies needed for build)
RUN npm install && npm cache clean --force

# Copy source code
COPY . .

# Build the app (REACT_APP_API_URL will be available during build)
RUN npm run build

# Production stage - minimal nginx image
FROM nginx:alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets
COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 8080

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8080/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
