import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { VeoModule } from './gemini/gemini.module';
import { DocumentTopicsModule } from './document-topics/document-topics.module';
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    VeoModule,
    DocumentTopicsModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
