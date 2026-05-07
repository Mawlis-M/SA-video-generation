import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

// Long pipelines (document-to-video can run 30+ minutes) make a single leaked promise
// rejection fatal under the Node 16+ default. Register handlers BEFORE bootstrap so we
// capture failures during startup too, and so the server never silently dies in the
// middle of a Veo job — leaving Postman with ECONNREFUSED and on-disk progress orphaned.
process.on('uncaughtException', (err) => {
  console.error(
    '[FATAL] uncaughtException — server kept alive so the running pipeline can finish or fail with a normal HTTP error.',
    err instanceof Error ? err.stack || err.message : err,
  );
});

process.on('unhandledRejection', (reason) => {
  console.error(
    '[FATAL] unhandledRejection — server kept alive so the running pipeline can finish or fail with a normal HTTP error.',
    reason instanceof Error ? reason.stack || reason.message : reason,
  );
});

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { logger: ['error', 'warn'] });
  app.enableCors({
    origin: [
      'http://localhost:5173',
      'http://localhost:8080',
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

bootstrap().catch((err) => {
  console.error('[FATAL] bootstrap failed:', err instanceof Error ? err.stack || err.message : err);
  process.exit(1);
});