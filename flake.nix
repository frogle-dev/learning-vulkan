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
          shaderc
          shader-slang
          sdl3
          glm
          vulkan-memory-allocator
          gdb
        ];
        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config gcc ];

        shellHook = ''
          export VK_LAYER_PATH=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d
        '';
      };
    };
}
