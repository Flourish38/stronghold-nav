function print_correct_path(stronghold)
    r = stronghold[end]
    while r.piece != 11
        println(r)
        r = stronghold[r.exits[r.targetDirection]]
    end
    println(r)
end