namespace QLearning.Avalonia

open Avalonia
open Avalonia.Controls
open Avalonia.Controls.ApplicationLifetimes
open Avalonia.Markup.Xaml
open QLearning.Avalonia.ViewModels
open QLearning.Avalonia.Views

type App() =
    inherit Application()

    override this.Initialize() =
            AvaloniaXamlLoader.Load(this)
            
    member this.Show(window: Window) =
        match this.ApplicationLifetime with
        | :? IClassicDesktopStyleApplicationLifetime as desktopLifetime ->
            desktopLifetime.MainWindow <- window
        | :? ISingleViewApplicationLifetime as singleViewLifetime ->
            singleViewLifetime.MainView <- window
        | _ -> ()

    override this.OnFrameworkInitializationCompleted() =
        // match this.ApplicationLifetime with
        // | :? IClassicDesktopStyleApplicationLifetime as desktopLifetime ->
        //     desktopLifetime.MainWindow <- MainWindow(DataContext = MainViewModel())
        // | :? ISingleViewApplicationLifetime as singleViewLifetime ->
        //     singleViewLifetime.MainView <- MainView(DataContext = MainViewModel())
        // | _ -> ()
        base.OnFrameworkInitializationCompleted()
