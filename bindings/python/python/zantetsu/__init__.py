"""Zantetsu - Fast anime metadata parser.

Extracts title, episode, resolution, codecs, and more from anime filenames.

Example:
    >>> from zantetsu import HeuristicParser
    >>> parser = HeuristicParser()
    >>> result = parser.parse("[SubsPlease] Spy x Family - 01 (1080p).mkv")
    >>> print(result.title)
    'Spy x Family'
    >>> print(result.episode)
    '1'
    >>> print(result.resolution)
    'FHD1080'
"""

from zantetsu._zantetsu import HeuristicParser, ParseResult

__version__ = "0.1.2"
__all__ = ["HeuristicParser", "ParseResult"]