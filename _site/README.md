ShinAsakawa.github.io

# 簡単な GitHub 管理方法解説

## 手順 1. `git clone` コマンドで GitHub の内容をすべてローカルディレクトリにダウンロードする

```bash
$ git clone https://ShinAsakawa.github.io.git [ダウンロードするディレクトリ名. 省略可]
$ cd ShinAsakawa.githuh.io.git [または上のコマンドで指定したディレクトリ名]
```

## 手順 2. ページの更新方法

1. ローカルファイルをチェックアウト checkout して更新作業を行い，
2. その内容を ``commit`` し
3. サーバ (GitHub) へ ``push`` する。

以下のサンプル操作を参照のこと

```bash
$ git checkout -b 適当なブランチ名

編集作業の実施

$ git commit -m '編集した内容' -a
$ git push -u origin ブランチ名
```

### 2.2 pull request の merge

1. GitHub 上で pull request を作成する
2. mergeは管理者が行う。merge 後はローカルで以下を実行し ``branch`` を削除する

```bash
$ git checkout master
$ git pull
$ git branch -d ブランチ名  [-d オプションは delete の意]
```

## 3. GitHub の 設定・使い方リンク

- [Getting started with GitHub](https://help.github.com/en/github/getting-started-with-github)
- [SSHによるGitHubへのアクセス](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)
- [pull requestの作成](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

## 4. local でページの見た目確認手順

GitHub pages は ``jeykell`` での管理を前提としている場合が多い。
``jeykell`` は静的ページの生成ツールである。日本語の文書も用意されている <http://jekyllrb-ja.github.io/> 

### 4.1 ``jekyll`` 操作のための command line tools のインストール

```bash
$ cd ShinAsaawa.github.io.git
$ gem install bundler
$ bundle init
$ vim Gemfile (他のテキストエディアでも可)

(Gemfileを開いて，一番下に次行を追記する)
gem "github-pages", group: :jekyll_plugins

$ bundle install
```

### 4.2 ``jekyll`` の起動

以下を実行するとローカルでサーバが立ち上がるので，ブラウザで http://localhost:4000/ にアクセスする。
または ``--port`` オプションで指定したポート番号にアクセスする。
ポート番号は設定ファイル ``_config.yml`` ファイル中に ``port:####`` のように指定することもできる。

```bash
$ bundle exec jekyll serve [--port ####]
```

## 5. リポジトリアクセス権限の修正

### 5.1. push で アクセス権限がないために起因するエラー 403 の回避方法 

自分のリポジトリを作ったアカウントのユーザー名とメールアドレスを登録していなかった場合、以下のコードで設定

```bash
$ git config --global user.name "<ユーザー名>"
$ git config --global user.email <メールアドレス>

(push する際に https://github.com/<ユーザ名>/<リポジトリ名>.git となっていた URL にユーザー名を入れる)
git remote set-url origin https://<ユーザ名>@github.com/ShinAsakawa/ShinAsakawa.github.io.git
```

- 参照: <https://hacknote.jp/archives/54105/>

## 6. git merge, git pull 時の不整合の解消

```bash
$ git fetch
```

でエラーがあって元に戻したい時は
ローカルの ``master``ブランチまで更新されていない（merge されていない）ので以下のコマンドを実行する

```bash
$ git reset --hard HEAD
```

これで直前の ``commit`` まで戻して、無かった事にできる。一方，

```bash
$ git pull
```

でエラーがあって元に戻したい場合，すなわちコンフリクトを無くしたいという時には
``pull = fetch + merge`` であるから ``merge`` を以下のコマンドで取り消す。

```bash
$ git merge --abort

```
その後は上の ``fetch`` のときと同じように以下のコマンドを実行する

```
$ git reset --hard HEAD
```

- 参照: https://qiita.com/wann/items/688bc17460a457104d7d
