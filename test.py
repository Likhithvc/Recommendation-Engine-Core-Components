from engine.orchestrator import RecommendationOrchestrator

engine = RecommendationOrchestrator()
print(engine.get_recommendations(1))