pipeline {
  agent any

  options {
    timestamps()
    timeout(time: 30, unit: 'MINUTES')
    buildDiscarder(logRotator(numToKeepStr: '20'))
  }

  // Poll SCM on push + scheduled weekly retrain (re-runs train + quality gate + deploy)
  triggers {
    pollSCM('H/5 * * * *')
    cron('H 2 * * 0')
  }

  environment {
    IMAGE    = 'mlmon-app'
    COMPOSE  = 'docker compose -p mlmon'
    MIN_ACCURACY = '0.85'
  }

  stages {
    stage('Setup') {
      steps {
        sh '''
          python3 -m venv .venv
          . .venv/bin/activate
          pip install -q -r requirements-dev.txt
        '''
      }
    }

    stage('Lint') {
      steps { sh '. .venv/bin/activate && flake8 --max-line-length 120 app.py train.py features.py tests scripts' }
    }

    stage('Train + Evaluate (quality gate)') {
      steps {
        sh '. .venv/bin/activate && python3 train.py | tee train.log'
        archiveArtifacts artifacts: 'artifacts/metrics.json, train.log', fingerprint: true
      }
    }

    stage('Test') {
      steps { sh '. .venv/bin/activate && pytest -q --junitxml=test-results.xml' }
      post { always { junit 'test-results.xml' } }
    }

    stage('Build image') {
      steps { sh 'docker build -t ${IMAGE}:${BUILD_NUMBER} -t ${IMAGE}:latest .' }
    }

    stage('Smoke test stack') {
      steps {
        sh '''
          ${COMPOSE} up -d --no-build app
          for i in $(seq 1 30); do
            curl -sf http://app:5000/health || curl -sf http://localhost:5000/health && break || sleep 2
          done
          curl -sf -X POST http://app:5000/api/predict \
            -H 'Content-Type: application/json' \
            -d '{"url":"https://www.google.com"}' | grep legitimate
          curl -sf http://app:5000/metrics | grep model_predictions_total
        '''
      }
    }

    stage('Deploy') {
      steps {
        sh '${COMPOSE} up -d'
      }
    }
  }

  post {
    failure { sh '${COMPOSE} logs --tail=100 || true' }
  }
}
