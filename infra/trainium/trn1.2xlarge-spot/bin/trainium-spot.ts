#!/usr/bin/env node
import * as dotenv from 'dotenv';
dotenv.config();
import * as cdk from 'aws-cdk-lib/core';
import { TrainiumSpotStack } from '../lib/trainium-spot-stack';

const app = new cdk.App();
new TrainiumSpotStack(app, 'TrainiumSpotStack', {
  env: {
    region: 'us-east-1',
    account: process.env.CDK_DEFAULT_ACCOUNT,
  },
  description: 'Cost-optimized Trainium1 spot instance with auto-shutdown on inactivity detection',
});
