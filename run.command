#!/bin/bash
# Синхронізатор XML для Premiere Pro
# Подвійний клік для запуску на Mac

APPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APPDIR"

# ─── Автооновлення з GitHub ──────────────────────────────────────────────────
REPO_RAW="https://raw.githubusercontent.com/mrkoss-seo/mooms_helper/main"

update_from_github() {
    echo "Перевіряю оновлення..."

    # Завантажуємо версію з GitHub (таймаут 5с)
    REMOTE_VERSION=$(curl -sL --connect-timeout 5 --max-time 10 "$REPO_RAW/VERSION" 2>/dev/null)

    # Якщо не вдалося — пропускаємо
    if [ -z "$REMOTE_VERSION" ]; then
        echo "  Не вдалося перевірити (немає інтернету?), пропускаю"
        return
    fi

    # Читаємо локальну версію
    LOCAL_VERSION="0"
    if [ -f "$APPDIR/VERSION" ]; then
        LOCAL_VERSION=$(cat "$APPDIR/VERSION" | tr -d '[:space:]')
    fi

    REMOTE_VERSION=$(echo "$REMOTE_VERSION" | tr -d '[:space:]')

    echo "  Локальна: $LOCAL_VERSION, GitHub: $REMOTE_VERSION"

    # Порівнюємо (числове порівняння через awk)
    NEED_UPDATE=$(awk "BEGIN { print ($REMOTE_VERSION > $LOCAL_VERSION) ? 1 : 0 }")

    if [ "$NEED_UPDATE" = "1" ]; then
        echo "  Знайдено оновлення! Завантажую..."

        # Завантажуємо нові файли у тимчасову папку
        TMPDIR=$(mktemp -d)

        # Список файлів для оновлення
        UPDATED=0
        for fname in sync_xml.py VERSION run.command requirements.txt; do
            HTTP_CODE=$(curl -sL --connect-timeout 5 --max-time 30 \
                -w "%{http_code}" -o "$TMPDIR/$fname" \
                "$REPO_RAW/$fname" 2>/dev/null)

            if [ "$HTTP_CODE" = "200" ] && [ -s "$TMPDIR/$fname" ]; then
                cp "$TMPDIR/$fname" "$APPDIR/$fname"
                echo "  ✓ $fname оновлено"
                UPDATED=$((UPDATED + 1))
            fi
        done

        rm -rf "$TMPDIR"

        if [ "$UPDATED" -gt 0 ]; then
            echo "  Оновлено $UPDATED файл(ів)!"

            # Якщо оновився сам run.command — перезапускаємо
            if [ -f "$APPDIR/run.command" ]; then
                chmod +x "$APPDIR/run.command"
            fi
        fi
    else
        echo "  Версія актуальна"
    fi
}

# Оновлення (не блокує запуск при помилках)
update_from_github

# ─── Шукаємо Python 3 з робочим tkinter ─────────────────────────────────────
# Системний 3.9 крашиться на нових macOS через старий Tcl/Tk
PY=""
for candidate in \
    /opt/homebrew/bin/python3 \
    /usr/local/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.*/bin/python3 \
    /usr/bin/python3; do

    for p in $candidate; do
        if [ -x "$p" ]; then
            if $p -c "import tkinter; root=tkinter.Tk(); root.destroy()" 2>/dev/null; then
                PY="$p"
                break 2
            fi
        fi
    done
done

if [ -z "$PY" ]; then
    echo "Не знайдено Python 3 з робочим tkinter!"
    echo ""
    echo "Встановіть Python 3 з https://www.python.org/downloads/"
    echo ""
    read -p "Натисніть Enter для закриття..."
    exit 1
fi

echo "Python: $PY"

# ─── Перевірка ffmpeg ─────────────────────────────────────────────────────────
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg не знайдено!"
    echo "Встановіть: brew install ffmpeg"
    echo ""
    read -p "Натисніть Enter для закриття..."
    exit 1
fi

# ─── Запуск (v5: без venv, без numpy/scipy!) ─────────────────────────────────
echo "Запускаю синхронізатор..."
nohup $PY "$APPDIR/sync_xml.py" &>/dev/null &
disown

osascript -e 'tell application "Terminal" to close front window' &>/dev/null &
