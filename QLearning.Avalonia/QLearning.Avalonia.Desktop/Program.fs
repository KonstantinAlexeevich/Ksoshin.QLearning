namespace QLearning.Avalonia.Desktop
open System
open Avalonia
open Gym.Environments
open Gym.Environments.Envs.Classic
open Gym.Rendering.Avalonia
open QLearning.Avalonia
open System.Threading.Tasks

module Program =

    [<CompiledName "BuildAvaloniaApp">] 
    let buildAvaloniaApp () = 
        AppBuilder
            .Configure<App>()
            .UsePlatformDetect()
            .AfterSetup(fun x ->
                
                let window = new AvaloniaEnvViewer(500, 500, "CartPole")
                let cartPole = new CartPoleEnv(new IEnvironmentViewerFactoryDelegate(
                    fun width -> fun height -> fun title -> Task.FromResult(window :> IEnvViewer)))

                cartPole.Reset() |> ignore
                cartPole.Render() |> ignore
                (x.Instance :?> App).Show(window)
                
                Task.Run(fun () -> QLearning.run(cartPole)) |> ignore)
            .LogToTrace(areas = Array.empty)

    [<EntryPoint; STAThread>]
    let main argv =
        buildAvaloniaApp().StartWithClassicDesktopLifetime(argv) |> ignore
        1