{
  description = "vulkan engine dev shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          vulkan-headers
          vulkan-loader
          vulkan-validation-layers
          vulkan-tools
          vulkan-tools-lunarg
          shaderc
          shader-slang
          sdl3
          llvm
          llvmPackages.compiler-rt
          glm
          vulkan-memory-allocator
          gdb
          renderdoc
        ];
        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config gcc ];

        shellHook = ''
          export VK_LAYER_PATH=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d
          export VK_ADD_LAYER_PATH=$VK_LAYER_PATH
          export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:${pkgs.renderdoc}/lib:$LD_LIBRARY_PATH
        '';
      };
    };
}
