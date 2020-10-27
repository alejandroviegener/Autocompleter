""" Template for autocomplete server, you probably won't need to edit anything
in this file. """

from flask import Flask, request, jsonify
import autocompleter

app = Flask(__name__)


@app.route('/autocomplete')
def autocomplete():
    """ Generate autocompletions given the query 'q' """
    q = request.args.get('q')
    completions = my_autocompleter.generate_completions(q)
    return jsonify({"Completions": completions})

if __name__ == "__main__":
    my_autocompleter = autocompleter.Autocompleter.load()

    app.run()
