from ai_mod1 import PocketProfessorAI


MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
prof = PocketProfessorAI(MODEL_PATH)

           
question = "i have two virtual machines with bridged networks why cant they ping eachother, they both have the same ip subnet and gateway"


hint, time_taken = prof.get_hint(question)


print(hint)


print(f"\nResponse time: {time_taken:.2f} seconds")
