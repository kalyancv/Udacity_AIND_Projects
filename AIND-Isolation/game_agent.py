"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import isolation
import itertools

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def weighted_heuristic_score(game, player):
    """The "Weighted" evaluation function discussed in lecture that outputs a
     score equal to the difference in the number of moves available to the
     two players. Weightage changed based on player center location.
     if player is center then play aggressive mode.
     if opponent player is center then play defensive mode.
     other cases less defensive mode.

     Parameters
     ----------
     game : `isolation.Board`
         An instance of `isolation.Board` encoding the current state of the
         game (e.g., player locations and blocked cells).

     player : hashable
         One of the objects registered by the game object as a valid player.
         (i.e., `player` should be either game.__player_1__ or
         game.__player_2__).

     Returns
     ----------
     float
         The heuristic value of the current game state
     """

    my_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    if game.is_loser(player) or len(my_moves) == 0:
        return float("-inf")

    if game.is_winner(player) or len(opp_moves) == 0:
        return float("inf")

    center = (int(game.width / 2), int(game.height / 2))

    player_location = game.get_player_location(player)
    opp_player_location = game.get_player_location(game.get_opponent(player))
    common_moves = set(my_moves).intersection(set(opp_moves))
    if player_location == center:
        return float(len(my_moves)  - 1.5 * len(opp_moves))+2.0 * len(common_moves)
    elif opp_player_location in center:
        return float(2.0 * len(my_moves) - 1.0 * len(opp_moves))+2.0 * len(common_moves)
    else:
        return 1.5 * len(my_moves) - 1.0 * len(opp_moves)+2.0 * len(common_moves)

def weighted_heuristic_score_2(game, player):
    """The "Weighted" evaluation function discussed in lecture that outputs a
     score equal to the difference in the number of moves available to the
     two players. Differnce between player and opponent player moves and added weightage for playing aggressive mode.


     Parameters
     ----------
     game : `isolation.Board`
         An instance of `isolation.Board` encoding the current state of the
         game (e.g., player locations and blocked cells).

     player : hashable
         One of the objects registered by the game object as a valid player.
         (i.e., `player` should be either game.__player_1__ or
         game.__player_2__).

     Returns
     ----------
     float
         The heuristic value of the current game state
     """

    my_moves =  game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    common_moves = set(my_moves).intersection(set(opp_moves))
    if game.is_loser(player) or len(my_moves) == 0:
        return float("-inf")

    if game.is_winner(player) or len(opp_moves) == 0:
        return float("inf")

    return float(1.25 * len(my_moves) - len(opp_moves) + len(common_moves))


def weighted_heuristic_score_3(game, player):  # formerly class CenterBias
    """The "Weighted" evaluation function discussed in lecture that outputs a
     score equal to the difference in the number of moves available to the
     two players. weightage changed based on player moves.
     if player have more moves then play aggressive mode.
     if opponent player have more moves then play defensive mode.

     Parameters
     ----------
     game : `isolation.Board`
         An instance of `isolation.Board` encoding the current state of the
         game (e.g., player locations and blocked cells).

     player : hashable
         One of the objects registered by the game object as a valid player.
         (i.e., `player` should be either game.__player_1__ or
         game.__player_2__).

     Returns
     ----------
     float
         The heuristic value of the current game state
     """

    my_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if game.is_loser(player) or my_moves == 0:
        return float("-inf")

    if game.is_winner(player) or opp_moves == 0:
        return float("inf")

    if(my_moves < opp_moves):
        return 1.0 * my_moves - 1.5 * opp_moves
    else:
        return 1.5 * my_moves - 1.0 * opp_moves

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    return weighted_heuristic_score(game, player)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    return weighted_heuristic_score_2(game, player)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    return weighted_heuristic_score_3(game, player)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # depth reached end
        _, best_move = self.get_min_max(game, depth)

        return best_move

    def get_min_max(self, game, depth, maximizing_player=True):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth <= 0:
            return self.score(game, self), game.get_player_location(self)

        # Get legal moves for active player
        legal_moves = game.get_legal_moves(self) if maximizing_player else game.get_legal_moves(game.get_opponent(self))
        location = game.get_player_location(self) if maximizing_player else game.get_player_location(game.get_opponent(self))
        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = (-1, -1)

        # Game over terminate
        if not legal_moves:
            # -inf or +inf from point of view of maximizing player
            return self.score(game, self), best_move

        for move in legal_moves:
            f_move = game.forecast_move(move)
            f_loc = f_move.get_player_location(self)
            score, _ = self.get_min_max(f_move, depth - 1, not maximizing_player)
            if maximizing_player:
                # Maximize the player
                if score > best_score:
                    best_score, best_move = score, move
            else:
                # Minimize the player
                if score < best_score:
                    best_score, best_move = score, move
        return best_score, best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            for depth in range(1, game.width * game.height):
                best_move = self.alphabeta(game, depth)
                #if best_score == float('inf'):
                #    break
        #    return best_move
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # depth reached end
        _, best_move = self.get_alphabeta(game, depth)

        return best_move


    def get_alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth <= 0:
            return self.score(game, self), game.get_player_location(self)

        # Get legal moves for active player
        legal_moves = game.get_legal_moves(self) if maximizing_player else game.get_legal_moves(game.get_opponent(self))
        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = (-1, -1)

        # Game over terminate
        if not legal_moves:
            # -inf or +inf from point of view of maximizing player
            return self.score(game, self), best_move

        for move in legal_moves:
            score, _ = self.get_alphabeta(game.forecast_move(move), depth - 1, alpha, beta, not maximizing_player)
            if maximizing_player:
            # Maximize the player
                if score > best_score:
                    best_score, best_move = score, move
                alpha = max(best_score, alpha)
                if best_score >= beta:
                    return best_score, best_move

            else:
            # Minimize the player
                if score <= best_score:
                    best_score, best_move = score, move
                beta = min(best_score, beta)
                if best_score <= alpha:
                    return best_score, best_move

        return best_score, best_move
