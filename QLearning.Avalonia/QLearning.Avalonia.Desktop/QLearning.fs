namespace QLearning.Avalonia.Desktop

open System
open System.Collections
open System.Collections.Generic
open Gym.Environments.Envs.Classic
open TorchSharp

module Array =
    let rand = new Random()
    let swap x y (a: 'a []) =
        let tmp = a.[x]
        a.[x] <- a.[y]
        a.[y] <- tmp
        
    let shuffle a =
        Array.iteri (fun i _ -> a |> swap i (rand.Next(i, Array.length a)))
        |> ignore
        a

module QLearning =
    
    type Transition(
        state: float array,
        action: int64,
        reward: float,
        next_state: float array,
        isDone: bool
        ) =
        member this.State = state
        member this.Action = action
        member this.Reward = reward
        member this.NextState = next_state
        member this.IsDone = isDone
        
    type ReplayMemory(capacity: int) =
        let memory = Queue<Transition>(capacity)

        member this.Push(transition) =
            memory.Enqueue(transition)
            while (memory.Count > capacity) do
                memory.Dequeue() |> ignore

        member this.Sample(batch_size: int) =
            memory
            |> Seq.toArray
            |> Array.shuffle
            |> Seq.take batch_size
            |> Seq.toArray
            
        member this.Length() = memory.Count
    
    type DQN(state_size: int, action_size: int) as this =
        inherit torch.nn.Module<torch.Tensor, torch.Tensor>("DQN")
        let fc1 = torch.nn.Linear(state_size, 32)
        let fc2 = torch.nn.Linear(32, 64)
        let fc3 = torch.nn.Linear(64, action_size)
        do
            this.RegisterComponents()
            
        override this.forward(x) =
            x
            |> fc1.forward
            |> torch.nn.ReLU().forward
            |> fc2.forward
            |> torch.nn.ReLU().forward
            |> fc3.forward
            
    type Agent(
        state_size: int,
        actions_size: int,
        learn_every: int,
        memory_size: int,
        sample_size: int,
        learning_rate: float,
        gamma_learn: float,
        tau_learn: float) =
        
        let policy_net = new DQN(state_size, actions_size)
        let target_net = new DQN(state_size, actions_size)
        let optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        let memory = ReplayMemory(memory_size)
        let mutable t_step = 0
        
        member this.PolicyNet() = policy_net

        member this.Step(transition) =
            memory.Push(transition)
            t_step <- (t_step + 1) % learn_every
            if memory.Length() > sample_size && t_step = 0 then
                let experiences = memory.Sample(sample_size)
                this.Learn(experiences)

        member this.Act(state: float array, eps: float) =
            
            let stateTensor = torch.tensor(state, torch.ScalarType.Float32)
            use _ = torch.no_grad()
            policy_net.eval()
            
            let action_values = stateTensor |> policy_net.forward
            
            policy_net.train()
            
            if Random().NextDouble() > eps then
                action_values.argmax().item<int64>() |> int32
            else
                Random().Next(0, actions_size)
            
        member this.Learn(transitions) =
            
            transitions
            |> Seq.iter (fun x ->
                let statesTensor = torch.tensor(x.State, torch.ScalarType.Float32)
                let actionsTensor = torch.tensor([ x.Action |> float ] |> Seq.toArray, torch.ScalarType.Float32)
                let nextStatesTensor = torch.tensor(x.NextState, torch.ScalarType.Float32)
                let rewardsTensor = torch.tensor([ x.Reward |> float ] |> Seq.toArray, torch.ScalarType.Float32)
                let donesTensor = torch.tensor([ x.IsDone |> (fun x -> if x then 1. else 0.) ] |> Seq.toArray, torch.ScalarType.Float32)
                
                let qTargetNext = target_net.forward(nextStatesTensor).max()
                let qTargets = rewardsTensor + (gamma_learn * qTargetNext) * (1 - donesTensor)
                let qExpected = policy_net.forward(statesTensor).gather(0, x.Action).unsqueeze(0)
                    
                let loss = torch.nn.functional.mse_loss(qExpected, qTargets, torch.nn.Reduction.Sum)    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() |> ignore
            )
            
            let local_params = policy_net.parameters() |> Seq.toList
            let target_params = target_net.parameters() |> Seq.toList
            Seq.zip local_params target_params
            |> Seq.iter (fun (local_param, target_param) ->
                target_param.requires_grad <- false
                target_param.copy_(tau_learn * local_param + (1.0 - tau_learn) * target_param) |> ignore
            )
            
    let run(env: CartPoleEnv) =
        
        let memory_size = 100
        let learn_every = 10
        let sample_size = 64
                
        let state_size = 4
        let actions_size = env.ActionSpace.Shape.Size
        let learning_rate = 0.01
        let gamma_learn = 0.99
        let tau_learn = 0.001
        
        let episodes_count = 2000
        
        let eps_start = 1.0
        let eps_end = 0.05
        let eps_decay = 0.995
        
        let agent = Agent(
            state_size = state_size,
            actions_size = actions_size,
            learn_every = learn_every,
            memory_size = memory_size,
            sample_size = sample_size,
            learning_rate = learning_rate,
            gamma_learn = gamma_learn,
            tau_learn = tau_learn
        )
        
        let scores = ResizeArray<float>()
        let scores_window = Queue<float>(100)
        let mutable eps = eps_start
        
        for i_episode = 1 to episodes_count do
            let mutable state = env.Reset().ToArray<float>()
            let mutable score = 0.
            let mutable isDone = false
            let mutable len = 0
                            
            while not isDone do
                let action = agent.Act(state, eps)
                let step = env.Step(action)
                let next_state = step.Observation.ToArray<float>()
                isDone <- step.Done
                
                let mutable reward = (float step.Reward)
                if isDone then
                    reward <- -100.0
                else
                    score <- (score + float reward)
                
                let transition = Transition(state, action, reward, next_state, isDone)
                agent.Step(transition)
                state <- next_state
                len <- len + 1
                
                env.Render() |> ignore
                
                if isDone then
                    ()
                    
            scores_window.Enqueue(score)
            scores.Add(score)
            if scores_window.Count > 100
                then scores_window.Dequeue() |> ignore
                
            eps <- max eps_end (eps_decay * eps)
            printfn $"\rEpisode %d{i_episode}\tAverage Score: %.2f{Seq.average scores_window} length: {len}"

            if i_episode % 100 = 0 then
                printfn "\rEpisode %d\tAverage Score: %.2f" i_episode (Seq.average scores_window)
