cd ./rust
cargo build --release
cd ..
mkdir -p ./lib/g_mind
cp ./rust/target/release/libg_mind.so lib/g_mind
cp -r ./godot/scripts ./lib/g_mind
touch ./g_mind.gdextension
cat <<EOL > ./g_mind.gdextension
[configuration]
entry_symbol = "gdext_rust_init"
compatibility_minimum = 4.2

[libraries]
linux.debug.x86_64 = "res://lib/g_mind/libg_mind.so"
linux.release.x86_64 = "res://lib/g_mind/libg_mind.so"
EOL
tar -cvf ./lib_g-mind.tar ./lib ./g_mind.gdextension
rm ./g_mind.gdextension -r ./lib
