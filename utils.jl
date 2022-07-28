print_room(env::StrongholdEnvironment) = print_room(env.stronghold, env.room)
print_room(stronghold, r) = print_room(stronghold, stronghold[r])
function print_room(stronghold, r::Room)
    println(
        _pieces_vec[r.piece], "\r\t\t  ",
        r.depth, "\t",
        _pieces_vec[get_piece(stronghold, r.parent)], "\r\t\t\t\t\t  ",
        [_pieces_vec[get_piece(stronghold, r.exits[i])] for i in 1:PIECE_TO_NUM_EXITS[r.piece]]
    )
end

function print_correct_path(stronghold)
    env = StrongholdEnvironment(stronghold, false)
    p = OraclePolicy(env)
    print_room(env)
    for (s, a, r, s′) in Episode(env, p)
        println(a, "\t", r)
        print_room(env)
    end
    return env.replay
end

function random_replay()
    env = StrongholdEnvironment(rand(strongholds), false)
    p = RandomPolicy()
    for _ in Episode(env, p); end
    return env.replay
end

begin
    struct FirstExitPolicy <: AbstractPolicy end
    Reinforce.action(π::FirstExitPolicy, r, s, A) = 1    
end

begin
    # Oracle policies need to have the environment to know which way to go, so need to be treated differently by functions.
    abstract type AbstractOraclePolicy <: AbstractPolicy end
    mutable struct OraclePolicy <: AbstractOraclePolicy env::Union{StrongholdEnvironment, Nothing} end
    Reinforce.action(π::OraclePolicy, r, s, A) = current(π.env).targetDirection
end