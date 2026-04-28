import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { logger: ['error', 'warn'] });
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
  await app.listen(port);

  const server = app.getHttpServer();
  server.setTimeout(0);
  server.keepAliveTimeout = 0;
  server.headersTimeout = 0;

  console.log('App listening on', port, '(no HTTP timeout)');
}
bootstrap();