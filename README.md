# CS3600TournamentBot
WE WILL BEAT CARRIE

# batch script commands
running a matchup batch (runs xy as x vs y and xy as y vs x)
python .\engine\batch_match_report.py run Bob Cassie --repeats 1

analyze all saved matches
python .\engine\batch_match_report.py analyze --pattern *.json

analyze all matches related to some bot
python .\engine\batch_match_report.py analyze --pattern AgentX.json