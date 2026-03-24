import { AppModule } from './app.module';
import { NestFactory } from '@nestjs/core';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { logger: false });
  
  app.enableCors({
    origin: [
      'http://localhost:5173', // Vite default
      'http://localhost:8080', // if your frontend really runs here
      'http://localhost:3001',
    ],
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: false,
  });
  
  const port = process.env.PORT ?? 3000;
  console.log('Environment variables:', port);
  console.log('Application was running on port', port);
  await app.listen(port);
}
bootstrap();