#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const testDir = path.join(root, 'test');
const caseFile = path.join(testDir, 'spec-test-cases.json');
const appFile = path.join(root, 'src', 'App.tsx');
const appLayoutFile = path.join(root, 'src', 'components', 'layout', 'AppLayout.tsx');

function fail(message) {
  console.error(`[FAIL] ${message}`);
  process.exitCode = 1;
}

if (!fs.existsSync(caseFile)) {
  fail(`Missing test case file: ${caseFile}`);
  process.exit(process.exitCode || 1);
}

const payload = JSON.parse(fs.readFileSync(caseFile, 'utf8'));
const cases = Array.isArray(payload.cases) ? payload.cases : [];

if (cases.length === 0) {
  fail('No test cases found in spec-test-cases.json');
}

if (fs.existsSync(appFile)) {
  const app = fs.readFileSync(appFile, 'utf8');
  const expectedRoutes = Array.isArray(payload.expected_routes) ? payload.expected_routes : [];
  for (const route of expectedRoutes) {
    if (!app.includes(`path="${route}"`) && !app.includes(`path='${route}'`) && !app.includes(`to="${route}"`) && !app.includes(`to='${route}'`)) {
      fail(`Route reference not found in src/App.tsx: ${route}`);
    }
  }
}

const highPriority = cases.filter((c) => c.priority === 'high').length;
console.log(`[OK] Loaded ${cases.length} cases (${highPriority} high priority).`);
if (process.exitCode && process.exitCode !== 0) {
  process.exit(process.exitCode);
}

if (cases.some((c) => /auth|signup|login|회원가입|로그인|내정보|mypage/i.test(c.title))) {
  if (!fs.existsSync(appLayoutFile)) {
    fail(`Missing auth layout file expected for auth scenarios: ${appLayoutFile}`);
  } else {
    const layout = fs.readFileSync(appLayoutFile, 'utf8');
    const hasAuthTabs = /회원가입|로그인/.test(layout);
    const hasMyPage = /내정보|\/mypage/.test(layout);
    const hasAuthConditional = /isAuthenticated\s*\?/.test(layout) && /:\s*\[\.{3}baseLinks|:\s*\[\.{3}baseLinks/.test(layout);

    if (!hasAuthTabs) {
      fail('Auth layout does not include signup/login tab state handling.');
    }
    if (!hasMyPage) {
      fail('Auth layout does not include mypage /내정보 expectation.');
    }
    if (!hasAuthConditional) {
      fail('Auth layout does not appear to switch links based on isAuthenticated.');
    }
  }
}
console.log('[OK] Spec smoke checks passed.');
