def print_board(board):1
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def winner(board, player):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]],
    ]
    return [player, player, player] in win_conditions

def get_empty_positions(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]

def minimax(board, depth, is_maximizing):
    if winner(board, "O"):
        return 1
    elif winner(board, "X"):
        return -1
    elif not get_empty_positions(board):
        return 0

    if is_maximizing:
        best_score = float("-inf")
        for (r, c) in get_empty_positions(board):
            board[r][c] = "O"
            score = minimax(board, depth + 1, False)
            board[r][c] = " "
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float("inf")
        for (r, c) in get_empty_positions(board):
            board[r][c] = "X"
            score = minimax(board, depth + 1, True)
            board[r][c] = " "
            best_score = min(score, best_score)
        return best_score

def best_move(board):
    best_score = float("-inf")
    move = None
    for (r, c) in get_empty_positions(board):
        board[r][c] = "O"
        score = minimax(board, 0, False)
        board[r][c] = " "
        if score > best_score:
            best_score = score
            move = (r, c)
    return move

def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"

    while True:
        print_board(board)
        if winner(board, "O"):
            print("O wins!")
            break
        elif winner(board, "X"):
            print("X wins!")
            break
        elif not get_empty_positions(board):
            print("It's a tie!")
            break

        if current_player == "X":
            row = int(input("Enter row: "))
            col = int(input("Enter col: "))
            if board[row][col] == " ":
                board[row][col] = "X"
                current_player = "O"
        else:
            print("AI is making a move...")
            (row, col) = best_move(board)
            board[row][col] = "O"
            current_player = "X"

if __name__ == "__main__":
    play_game()
