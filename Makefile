MONITORING_COMPOSE=docker-compose.prod.yml

.PHONY: monitoring-up monitoring-down monitoring-update

monitoring-up:
	docker-compose -f $(MONITORING_COMPOSE) up -d

monitoring-down:
	docker-compose -f $(MONITORING_COMPOSE) down

monitoring-update:
	docker-compose -f $(MONITORING_COMPOSE) pull
	docker-compose -f $(MONITORING_COMPOSE) up -d --remove-orphans
