import autocompleter
import sys


def build():
    """Build an autocompleter using the operator messages from the company with ID 50001"""
    my_autocompleter = autocompleter.Autocompleter()
    my_autocompleter.fit("sample_conversations.json", company_group_id_filter=[50001], verbose=True)
    my_autocompleter.save()

if __name__ == "__main__":
    sys.exit(build())
