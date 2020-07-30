# ShinAsakawa.github.io

## 1. git clone で GitHub の内容を download

```bash
$ git clone https://ShinAsakawa.github.io.git
$ cd ShinAsakawa.githuh.io.git
```
## 2. ページの更新方法

### 2.1 checkout

```bash
$ git checkout -b 適当なブランチ名

編集作業の実施

$ git commit -m '編集した内容' -a
$ git push -u origin ブランチ名
```

### 2.2 pull request の merge

1. GitHub上で pull request を作成する
2. mergeは管理者が行います．merge 後はローカルで以下実行

```bash
$ git checkout master
$ git pull
$ git branch -d ブランチ名
```

## 3. GitHub の 設定・使い方

- [Getting started with GitHub](https://help.github.com/en/github/getting-started-with-github)
- [SSHによるGitHubへのアクセス](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)
- [pull requestの作成](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

## 4. local でページの見た目確認手順

### 4.1 command line toolsのインストール

```bash
$ cd project-ccap.github.io.git
$ gem install bundler
$ bundle init
$ vim Gemfile (他のテキストエディアでも可)

(Gemfileを開いて，一番下に次行を追記する)
gem "github-pages", group: :jekyll_plugins

$ bundle install
```

以下を実行するとローカルでサーバが立ち上がるので，ブラウザでhttp://localhost:4000/ にアクセスする．

```bash
$ bundle exec jekyll serve
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

